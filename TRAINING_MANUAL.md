# LPRNet Training Manual

## Prerequisites

| Requirement | Tested Version |
|-------------|---------------|
| Python | 3.10 |
| PyTorch | 2.6.0+cu126 |
| OpenCV | 4.10.0 |
| imutils | 0.5.4 |
| NumPy | (bundled with PyTorch) |

**Python path used on this machine:**

```
C:\Users\mukil\AppData\Local\Programs\Python\Python310\python.exe
```

Install missing packages:

```bash
pip install torch torchvision opencv-python imutils numpy
```

---

## Data Preparation

### Directory layout

```
LPR-custom/
├── data/
│   ├── train/          ← training images
│   │   ├── TN01AB5274.jpg
│   │   ├── TN01AC0163.jpg
│   │   └── ...
│   ├── test/           ← test images (optional, same format)
│   └── load_data.py
├── model/
│   └── LPRNet.py
├── weights/            ← saved checkpoints (auto-created)
├── train_LPRNet.py
└── test_LPRNet.py
```

### Image labelling convention

The **filename** (without extension) **is** the label.

```
TN01AB5274.jpg  →  label = "TN01AB5274"
```

- Characters before the first `-` or `_` are used as the label.
- All characters must exist in the vocabulary (`0-9`, `A-Z`).
- Images can be any size — they are resized to **94 × 24** automatically.

### Current character set

Defined in `data/load_data.py`:

```
0-9  (10 digits)
A-Z  (26 letters)
-    (CTC blank token, always last)
```

Total classes: **37** (36 characters + 1 blank)

---

## Training

### Basic command

```bash
python train_LPRNet.py \
    --train_img_dirs data/train \
    --test_img_dirs data/train \
    --lpr_max_len 10 \
    --batch_size 16 \
    --max_iter 5000 \
    --augment
```

### Full flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--train_img_dirs` | `~/workspace/trainMixLPR` | Comma-separated paths to training image directories |
| `--test_img_dirs` | `~/workspace/testMixLPR` | Comma-separated paths to test image directories |
| `--img_size` | `[94, 24]` | Input image size `[width, height]` |
| `--lpr_max_len` | `8` | Maximum label length (**set to 10** for Indian plates) |
| `--batch_size` | `32` | Training batch size |
| `--test_batch_size` | `32` | Test batch size |
| `--max_iter` | `250000` | Total training iterations |
| `--learning_rate` | `0.001` | Initial learning rate (Adam) |
| `--lr_step` | `100000` | Drop LR by 10× every N iterations |
| `--gradient_noise` | `0.001` | Gradient noise scale |
| `--dropout_rate` | `0.5` | Dropout rate |
| `--augment` | off | Enable random affine augmentation |
| `--use_stn` | off | Enable Spatial Transformer Network |
| `--stn_warmup` | `5000` | Iterations before activating STN |
| `--cuda` | `True` | Use GPU (set `False` for CPU) |
| `--resume_iter` | `0` | Resume training from this iteration |
| `--pretrained_model` | `""` | Path to pretrained `.pth` weights |
| `--save_interval` | `5000` | Save checkpoint every N iterations |
| `--test_interval` | `5000` | Evaluate every N iterations |
| `--save_folder` | `./weights/` | Directory for saved weights |
| `--num_workers` | `8` | DataLoader workers |

### Recommended settings for this dataset (140 images)

```bash
python train_LPRNet.py \
    --train_img_dirs data/train \
    --test_img_dirs data/train \
    --lpr_max_len 10 \
    --batch_size 16 \
    --max_iter 5000 \
    --lr_step 2000 \
    --learning_rate 0.001 \
    --save_interval 1000 \
    --test_interval 1000 \
    --augment \
    --save_folder ./weights/
```

> With batch_size=16 and 140 images, one pass through the data ≈ 9 iterations.
> 5 000 iterations ≈ 555 epochs — sufficient for a small dataset.

### GPU vs CPU

```bash
# GPU (default)
python train_LPRNet.py ...

# CPU only
python train_LPRNet.py ... --cuda False
```

### Resume training

```bash
python train_LPRNet.py \
    --pretrained_model ./weights/LPRNet_iter_3000.pth \
    --resume_iter 3000 \
    ...
```

---

## Testing / Evaluation

### Greedy decoding (default)

```bash
python test_LPRNet.py \
    --test_img_dirs data/train \
    --lpr_max_len 10 \
    --pretrained_model ./weights/Final_LPRNet_model.pth
```

### Beam search decoding

```bash
python test_LPRNet.py \
    --test_img_dirs data/train \
    --lpr_max_len 10 \
    --pretrained_model ./weights/Final_LPRNet_model.pth \
    --decode_method beam_search \
    --beam_width 10
```

### Show predictions visually

```bash
python test_LPRNet.py \
    --test_img_dirs data/train \
    --pretrained_model ./weights/Final_LPRNet_model.pth \
    --show True
```

### Test flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--test_img_dirs` | `./data/test` | Test image directories |
| `--pretrained_model` | `./weights/Final_LPRNet_model.pth` | Trained model path |
| `--decode_method` | `greedy` | `greedy` or `beam_search` |
| `--beam_width` | `10` | Beam width (only for beam_search) |
| `--show` | `False` | Display each prediction in a window |
| `--use_stn` | off | Must match training flag |
| `--dropout_rate` | `0` | Keep at 0 for inference |
| `--cuda` | `True` | GPU / CPU toggle |

---

## Output Files

| Path | Description |
|------|-------------|
| `weights/LPRNet_iter_N.pth` | Checkpoint at iteration N |
| `weights/Final_LPRNet_model.pth` | Final model after training completes |

---

## Adding Your Own Data

1. **Collect** cropped license plate images (any size).
2. **Name** each file with its plate text: `MH12DE1234.jpg`.
3. **Place** images in a folder (e.g., `data/train/`, `data/test/`).
4. **Verify** all characters in filenames exist in `CHARS` inside `data/load_data.py`.
5. **Set** `--lpr_max_len` to the longest label length in your dataset.
6. **Train** using the command above.

### Extending the character set

Edit `CHARS` in `data/load_data.py` — keep `'-'` (blank) as the **last** entry:

```python
CHARS = [
    '0', '1', ..., '9',
    'A', 'B', ..., 'Z',
    # add new characters here
    '-'   # CTC blank — must stay last
]
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'imutils'` | `pip install imutils` |
| `Character 'x' not in CHARS vocabulary` | Add the character to `CHARS` in `data/load_data.py` |
| `CUDA out of memory` | Reduce `--batch_size` |
| Loss stays at `inf` | Check images are valid, labels match filenames |
| DLL load error on Windows | Use the correct Python: `C:\Users\mukil\AppData\Local\Programs\Python\Python310\python.exe` |
| Low accuracy on small dataset | Enable `--augment`, increase `--max_iter` |
