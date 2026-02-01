import torch
from torchvision import transforms
from PIL import Image
from resnet_model import ResNet18

def inference(model, input_tensor, device, class_names=None):
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

def load_model(checkpoint_path, device, num_classes=2):
    model = ResNet18(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    return model

def main():
    # Path to your checkpoint
    checkpoint_path = 'lightning_logs/version_0/checkpoints/epoch=9-step=140.ckpt'  # Update as needed
    class_names = ['cracked', 'non-cracked']  # Update as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device, num_classes=len(class_names))

    # Path to your image
    image_path = 'data/image.png'  # Update as needed
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    input_tensor = transform(img)

    pred_idx, pred_label = inference(model, input_tensor, device, class_names)
    print(f'Predicted class index: {pred_idx}, label: {pred_label}')

if __name__ == '__main__':
    main()
