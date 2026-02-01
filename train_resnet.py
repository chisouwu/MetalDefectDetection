import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
class SteelDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Determine columns for image id and label (robust to different CSV formats)
        if 'ImageId' in self.data.columns:
            self.img_col = 'ImageId'
        else:
            self.img_col = self.data.columns[0]
        if 'ClassId' in self.data.columns:
            self.label_col = 'ClassId'
        else:
            self.label_col = self.data.columns[1]

        # Filter out rows where the image file does not exist
        def _exists(fname):
            return os.path.exists(os.path.join(self.root_dir, fname))
        mask = self.data[self.img_col].apply(_exists)
        missing = (~mask).sum()
        if missing > 0:
            print(f"Warning: {missing} images listed in {csv_file} not found in {root_dir}. These will be skipped.")
            print(self.data.loc[~mask, self.img_col].head(10).tolist())
        self.data = self.data.loc[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = os.path.join(self.root_dir, row[self.img_col])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Unable to open image {img_name}: {e}")
        label = int(row[self.label_col]) - 1  # Convert 1-based CSV labels to 0-based
        if self.transform:
            image = self.transform(image)
        return image, label
def inference(model, input_tensor, device, class_names=None):
    """
    Perform inference on a single image tensor.
    Args:
        model: Trained model.
        input_tensor: A torch tensor of shape (C, H, W) or (1, C, H, W).
        device: torch.device.
        class_names: Optional list of class names.
    Returns:
        Predicted class index (and class name if provided).
    """
    model.eval()
    with torch.no_grad():
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
        if class_names:
            return pred_idx, class_names[pred_idx]
        return pred_idx
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet_model import ResNet18

def main():
    # Hyperparameters
    num_epochs = 4
    batch_size = 32
    learning_rate = 0.1
    num_classes = 4

    # Data
    # Patch training settings
    target_height = 256
    patch_width = 512
    use_patches = True  # set False to train on full-width resized images

    # Helper: resize to target height while keeping aspect ratio
    def resize_height(img, target_h=target_height):
        w, h = img.size
        new_w = int(w * (target_h / h))
        return img.resize((new_w, target_h), Image.BILINEAR)

    if use_patches:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: resize_height(img, target_height)),
            transforms.RandomCrop((target_height, patch_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: resize_height(img, target_height)),
            transforms.CenterCrop((target_height, patch_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # Reduce batch size for larger patches to avoid OOM
        if batch_size > 16:
            batch_size = 16
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: resize_height(img, target_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        val_transform = transform

    data_csv = './data/steel/train.csv'
    data_img_dir = './data/steel/train_images'  # Update if images are in a different folder
    full_dataset = SteelDataset(csv_file=data_csv, root_dir=data_img_dir, transform=transform)

    # If you have a separate test CSV, use it; otherwise split the dataset into train/val
    test_csv = './data/steel/test.csv'
    if os.path.exists(test_csv):
        test_img_dir = './data/steel/test_images'
        testset = SteelDataset(csv_file=test_csv, root_dir=test_img_dir, transform=val_transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        trainset = full_dataset
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        # Split full dataset into 90% train, 10% val (random, deterministic)
        val_fraction = 0.1
        total_len = len(full_dataset)
        val_len = max(1, int(total_len * val_fraction))
        train_len = total_len - val_len
        torch.manual_seed(42)
        perm = torch.randperm(total_len).tolist()
        train_indices = perm[:train_len]
        val_indices = perm[train_len:]
        trainset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = SteelDataset(csv_file=data_csv, root_dir=data_img_dir, transform=val_transform)
        testset = torch.utils.data.Subset(val_dataset, val_indices)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # If testloader wasn't defined earlier (e.g., when using separate test CSV), ensure trainloader exists
    if 'trainloader' not in locals():
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # Quick debug: set to True to inspect one training batch (shapes, ranges)
    debug = False
    if debug:
        batch = next(iter(trainloader))
        inputs, labels = batch
        print(f"Example batch - inputs shape: {inputs.shape}, dtype: {inputs.dtype}, min/max: {inputs.min().item()}/{inputs.max().item()}")
        print(f"labels shape: {labels.shape}, labels sample: {labels[:10]}")
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # === Resume training / checkpointing settings ===
    # By default training restarts from scratch. Set RESUME=True to resume from a checkpoint.
    CHECKPOINT_PATH = None  # optional explicit path to checkpoint; None = auto-find in saved_models
    RESUME = True  # set True to load weights and continue training; default is restart from scratch
    EXTRA_EPOCHS = 4  # number of additional epochs to run when resuming (ignored if RESUME=False)
    SAVE_DIR = 'checkpoints'
    SAVED_MODELS_DIR = 'saved_models'
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    start_epoch = 0
    original_num_epochs = num_epochs

    # If RESUME requested, try to auto-find latest checkpoint in saved_models (if CHECKPOINT_PATH not provided)
    if RESUME:
        chosen_ckpt = None
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            chosen_ckpt = CHECKPOINT_PATH
        else:
            # Search saved_models for files named like resnet_steel_epoch{N}.pth and pick the largest N
            import glob, re
            pattern = os.path.join(SAVED_MODELS_DIR, 'resnet_steel_epoch*.pth')
            candidates = glob.glob(pattern)
            max_epoch = -1
            best = None
            for p in candidates:
                m = re.search(r'resnet_steel_epoch(\d+)\.pth$', p)
                if m:
                    e = int(m.group(1))
                    if e > max_epoch:
                        max_epoch = e
                        best = p
            if best:
                chosen_ckpt = best
        # Fallback to lightning checkpoint if nothing in saved_models
        if chosen_ckpt is None and CHECKPOINT_PATH is None:
            # keep compatibility with previous default
            default_lightning = 'lightning_logs/version_0/checkpoints/epoch=9-step=140.ckpt'
            if os.path.exists(default_lightning):
                chosen_ckpt = default_lightning

        if chosen_ckpt and os.path.exists(chosen_ckpt):
            ckpt = torch.load(chosen_ckpt, map_location=device)
            # Handle checkpoint formats
            if isinstance(ckpt, dict):
                # typical saved_models format includes 'model_state_dict'
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                elif 'state_dict' in ckpt:
                    sd = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
                    model.load_state_dict(sd, strict=False)
                else:
                    # maybe direct state_dict
                    try:
                        model.load_state_dict(ckpt, strict=False)
                    except Exception:
                        print("Warning: Couldn't interpret checkpoint contents exactly; attempting best-effort load.")
                # epoch and optimizer
                if 'epoch' in ckpt:
                    start_epoch = int(ckpt['epoch'])
                else:
                    m = re.search(r'epoch=(\d+)', chosen_ckpt)
                    if m:
                        start_epoch = int(m.group(1))
                if 'optimizer_state_dict' in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    except Exception:
                        pass
            else:
                # ckpt might be a raw state_dict
                try:
                    model.load_state_dict(ckpt)
                except Exception as e:
                    print(f"Warning: failed to load checkpoint weights directly: {e}")
            print(f"Resuming from checkpoint {chosen_ckpt}, starting at epoch {start_epoch}")
            # If EXTRA_EPOCHS is set, extend num_epochs to run EXTRA_EPOCHS more epochs
            if EXTRA_EPOCHS and EXTRA_EPOCHS > 0:
                num_epochs = start_epoch + EXTRA_EPOCHS
            else:
                num_epochs = original_num_epochs
        else:
            print("RESUME requested but no checkpoint found in saved_models or default locations. Starting from scratch.")
    else:
        print("Starting training from scratch (not resuming checkpoint).")
    # ================================================

    # Training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Show current batch loss and running average loss in tqdm bar
            progress_bar.set_postfix({
                'batch_loss': loss.item(),
                'avg_loss': running_loss / (i+1)
            })

    # Save final model to dedicated directory
    FINAL_SAVE_DIR = 'saved_models'
    os.makedirs(FINAL_SAVE_DIR, exist_ok=True)
    final_epoch = num_epochs
    final_path = os.path.join(FINAL_SAVE_DIR, f'resnet_steel_epoch{final_epoch}.pth')
    torch.save({
        'epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes,
    }, final_path)
    print(f'Final model saved to: {final_path}')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    main()
