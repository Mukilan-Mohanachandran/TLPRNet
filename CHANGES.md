# LPRNet — Paper-Faithful Implementation Changes

Reference paper: **LPRNet: License Plate Recognition via Deep Neural Networks** ([arXiv:1806.10447](https://arxiv.org/abs/1806.10447))

---

## 1. Architecture Changes (`model/LPRNet.py`)

### 1.1 SmallBasicBlock (Paper Table 2)

| Aspect | Before | After (Paper) |
|--------|--------|---------------|
| Normalization | ReLU only after each conv | **BatchNorm + ReLU** after each conv (Sec. 3.1) |

### 1.2 Backbone Network (Paper Table 3)

| Aspect | Before | After (Paper) |
|--------|--------|---------------|
| Conv kernel for height collapse | `(1, 4)` — collapses width | **`(4, 1)`** — collapses height |
| Conv kernel for character context | `(13, 1)` — wide across height | **`(1, 13)`** — wide across width (Sec. 3.1: "1×13 kernel utilizes the local character context") |
| MaxPool3d stride (pool 2) | `(2, 1, 2)` — H preserved, W halved | **`(2, 2, 1)`** — H halved, W preserved |
| MaxPool3d stride (pool 3) | `(4, 1, 2)` — H preserved, W halved | **`(4, 2, 1)`** — H halved, W preserved |
| BN+ReLU after final class conv | Present | **Removed** (not in Table 3; applying ReLU on logits is undesirable for CTC) |
| Redundant BN+ReLU between SBBs | External BN+ReLU after each SBB | **Removed** (now internal to each SBB) |
| Output sequence length (T) | ~18 (due to width pooling) | **~74** (width preserved, per paper design) |

### 1.3 Global Context Embedding (Sec. 3.1, ref [12] ParseNet)

| Aspect | Before | After (Paper) |
|--------|--------|---------------|
| Method | Multi-scale fusion: collect features from 4 intermediate layers, custom power-normalization, concat, 1×1 conv | **ParseNet-style**: global average pool over backbone output → FC(128) + ReLU → tile to spatial size → concat with backbone output → 1×1 conv |

### 1.4 Spatial Transformer Network (Paper Table 1, optional)

| Aspect | Before | After (Paper) |
|--------|--------|---------------|
| STN / LocNet | Not implemented | **Added**: AvgPool → two parallel conv branches (stride 3 & 5) → concat → Dropout → FC(32, TanH) → FC(6, scaled TanH) → affine grid sample |
| STN warmup | N/A | **5k iterations** before enabling (Sec. 3.2) |

Enable with `--use_stn` flag.

---

## 2. Training Changes (`train_LPRNet.py`)

| Aspect | Before | After (Paper Sec. 3.2) |
|--------|--------|------------------------|
| Optimizer | RMSprop (lr=0.1, momentum=0.9, weight_decay=2e-5) | **Adam** (lr=0.001) |
| Batch size | 128 | **32** |
| Training duration | 15 epochs | **250,000 iterations** |
| LR schedule | Epoch-based `[4, 8, 12, 14, 16]`, factor 0.1 | **Step every 100k iterations**, factor 0.1 |
| Gradient noise | None | **Scale 0.001** added after backward pass |
| T_length (CTC input length) | Hardcoded to 18 | **Computed dynamically** from model output |
| Data augmentation | None | **Random affine** (rotation ±5°, scale 0.9–1.1, translation ±3px) via `--augment` flag |
| `build_lprnet()` call | `phase=True/False` (buggy: always evaluated as eval) | **`training=True/False`** (correct train/eval mode) |

---

## 3. Testing / Decoding Changes (`test_LPRNet.py`)

| Aspect | Before | After (Paper Sec. 3.1) |
|--------|--------|------------------------|
| Greedy decode | Custom implementation (functional but verbose) | **Cleaned up**: collapse consecutive duplicates → remove blanks |
| Beam search | Not implemented | **Added**: CTC prefix beam search (`--decode_method beam_search --beam_width 10`) |
| Post-filtering | Not implemented | Not implemented (requires country-specific plate templates; noted as future work) |

---

## 4. Data Pipeline Changes (`data/load_data.py`)

| Aspect | Before | After |
|--------|--------|-------|
| Augmentation | None | **Random affine transform** (rotation, scaling, translation) controlled by `augment=True/False` parameter |
| Label dtype | `np.int` (deprecated) | **`np.int64`** |

---

## 5. Interface Changes

### `build_lprnet()` — new signature

```python
# Old
build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5)

# New
build_lprnet(class_num=66, dropout_rate=0.5, use_stn=False, training=True)
```

- `lpr_max_len` removed (not used by the model)
- `phase` replaced by `training` (bool, correctly sets train/eval mode)
- `use_stn` added (enables optional Spatial Transformer Network)

### `LPRDataLoader` — new parameter

```python
# Old
LPRDataLoader(img_dir, imgSize, lpr_max_len, PreprocFun=None)

# New
LPRDataLoader(img_dir, imgSize, lpr_max_len, augment=False, PreprocFun=None)
```

### CLI flags — training

| New Flag | Default | Description |
|----------|---------|-------------|
| `--max_iter` | 250000 | Total training iterations |
| `--lr_step` | 100000 | LR decay interval |
| `--gradient_noise` | 0.001 | Gradient noise scale |
| `--use_stn` | off | Enable Spatial Transformer |
| `--stn_warmup` | 5000 | Iterations before STN activates |
| `--augment` | off | Enable random affine augmentation |
| `--batch_size` | 32 | Training batch size |

### CLI flags — testing

| New Flag | Default | Description |
|----------|---------|-------------|
| `--use_stn` | off | Must match training |
| `--decode_method` | greedy | `greedy` or `beam_search` |
| `--beam_width` | 10 | Beam width for beam search |

---

## 6. Breaking Changes

- **Model weights from the previous implementation are NOT compatible** with the new architecture (different layer shapes, channel counts, and global context structure).
- Training is now **iteration-based** instead of epoch-based.
- The `build_lprnet()` function signature has changed.

---

## 7. What Is NOT Implemented

| Paper Feature | Status | Reason |
|---------------|--------|--------|
| Post-filtering with LP templates | Not implemented | Country-specific; requires a template rule set per plate format |
| FPGA / OpenVINO deployment | Not implemented | Deployment-specific; out of scope for training code |

---

## 8. Dimensional Flow (New Architecture)

```
Input:          N ×   3 × 24 × 94
Conv 3×3 s1:    N ×  64 × 22 × 92
MaxPool3d s1:   N ×  64 × 20 × 90
SBB #128:       N × 128 × 20 × 90
MaxPool3d:      N ×  64 ×  9 × 88     (ch/2, H/2, W preserved)
SBB #256:       N × 256 ×  9 × 88
SBB #256:       N × 256 ×  9 × 88
MaxPool3d:      N ×  64 ×  4 × 86     (ch/4, H/2, W preserved)
Conv 4×1:       N × 256 ×  1 × 86     (height collapsed)
Conv 1×13:      N ×  Nc ×  1 × 74     (wide character-context conv)
Global ctx:     N ×  Nc ×  1 × 74     (FC→tile→concat→1×1 conv)
Squeeze:        N ×  Nc × 74          → T_length = 74
```

Where `Nc = class_num` (67 for Chinese plates including blank).
