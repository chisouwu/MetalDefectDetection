# ResNet Steel Classifier ✅

A small PyTorch project for classifying steel surface images into 4 classes using a ResNet18 model (custom implementation in `resnet_model.py`). The dataset is expected in `data/steel` with `train.csv` containing ImageId / ClassId and images in `data/steel/train_images`.

---

## Quick start 🔧

Requirements:
- Python 3.8+
- PyTorch
- torchvision
- pandas
- Pillow
- tqdm

Install (example):
```bash
pip install torch torchvision pandas pillow tqdm
```

Run training (default settings in `train_resnet.py`):
```bash
python train_resnet.py
```

Key settings (edit in `train_resnet.py`):
- `num_epochs`, `batch_size`, `learning_rate` (hyperparams)
- `num_classes = 4` for this dataset
- `use_patches`, `target_height`, `patch_width` — patch-based training preserves panoramic resolution (images ~1600x256)
- `RESUME`, `CHECKPOINT_PATH`, `EXTRA_EPOCHS` — control resuming from checkpoints

To inspect a single batch quickly, set `debug = True` in `train_resnet.py`.

---

## Inference & Evaluation 🧪

- `infer_resnet.py`: example script to load an image and run inference. Update `checkpoint_path`, `class_names`, and `image_path` before running.
- `evaluate_resnet.py` (if present): evaluate a saved checkpoint on `data/split/test` or other test sets.

Usage (example):
```bash
python infer_resnet.py
```

---

## Checkpoints & Saving 💾

- Final models are saved to `saved_models/` as `resnet_steel_epoch{N}.pth` by default.
- `RESUME=True` will try to pick the latest `resnet_steel_epoch{N}.pth` from `saved_models/` (or fall back to the Lightning checkpoint path if none found).

Note: `saved_models/`, `checkpoints/`, and `lightning_logs/` are ignored by `.gitignore` by default. If you want to include them in the repo, update `.gitignore` accordingly.

---

## Data

- Primary CSV: `data/steel/train.csv` (columns: `ImageId,ClassId,EncodedPixels`)
- Images directory: `data/steel/train_images/`
- `SteelDataset` reads the CSV and maps 1-based ClassId -> 0-based labels for PyTorch.

If some images are missing, the dataset currently warns and skips those entries.

---

## Git / Repo

- A `.gitignore` exists and excludes large artifacts by default. If you want to include `saved_models/` or `lightning_logs/`, remove those lines from `.gitignore` and add them to the repo.
- To initialize and push to GitHub:
```bash
git init
git add .
git commit -m "Initial import"
# add remote and push as desired
```

If you have large model files, consider using Git LFS.

---

If you want, I can:
- Add a CLI (`argparse`) to control resume/restart and hyperparameters,
- Add sliding-window inference to aggregate patch predictions,
- Add mixed-precision (AMP) training to save memory.

Tell me which next step you'd like. 🚀
