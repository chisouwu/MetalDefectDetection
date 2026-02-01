import os
import shutil
import random

def split_dataset(source_dir, cracked_dir, non_cracked_dir, output_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    for category, src in [('cracked', cracked_dir), ('non-cracked', non_cracked_dir)]:
        images = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        for split, split_images in [('train', train_images), ('test', test_images)]:
            split_path = os.path.join(output_dir, split, category)
            os.makedirs(split_path, exist_ok=True)
            for img in split_images:
                shutil.copy2(os.path.join(src, img), os.path.join(split_path, img))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'data')
    cracked_dir = os.path.join(source_dir, 'cracked')
    non_cracked_dir = os.path.join(source_dir, 'non-cracked')
    output_dir = os.path.join(source_dir, 'split')
    split_dataset(source_dir, cracked_dir, non_cracked_dir, output_dir)
    print("Dataset split complete. Train and test folders created in 'data/split'.")
