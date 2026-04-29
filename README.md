# TLPRNet

Research repository for architecturally combining ideas from **LPRNet** and **PaddleOCR** for license plate recognition.

## Overview

This project explores **architecture-level fusion**, not plain model ensembling:
- adapting and merging architectural components inspired by **LPRNet** and **PaddleOCR**
- building a unified recognition network design for robust plate text prediction
- evaluating how fused design choices improve recognition on real-world plate images

Current repo contents are focused on the LPRNet training/evaluation baseline, which serves as the starting point for fusion architecture experiments.

## Repository Structure

- `model/` - LPRNet model definition
- `data/load_data.py` - dataset loader and character vocabulary
- `train_LPRNet.py` - training script
- `test_LPRNet.py` - inference and evaluation script
- `weights/` - saved checkpoints and final model weights
- `TRAINING_MANUAL.md` - detailed setup and training options

## Quick Start

Install dependencies:

```bash
pip install torch torchvision opencv-python imutils numpy
```

Train:

```bash
python train_LPRNet.py --train_img_dirs data/train --test_img_dirs data/test --augment
```

Evaluate:

```bash
python test_LPRNet.py --test_img_dirs data/test --pretrained_model ./weights/Final_LPRNet_model.pth
```

## Data Format

- Each image filename (without extension) is treated as the label.
- Example: `TN01AB5274.jpg` -> label `TN01AB5274`
- Supported vocabulary is defined in `data/load_data.py`.

## Research Status

- LPRNet baseline training and decoding (greedy + beam search) is available.
- Architecture fusion experiments (LPRNet + PaddleOCR-inspired design) are in progress.