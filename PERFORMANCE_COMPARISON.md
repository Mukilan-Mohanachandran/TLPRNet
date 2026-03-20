# Backbone Performance Comparison

This document compares the original `lprnet` backbone with the new incremental `svtr_lcnet` hybrid backbone.

## Test Setup

- Input tensor: `batch=32`, shape `(N, 3, 24, 94)`
- Classes: `37` (`0-9`, `A-Z`, blank)
- Model mode: `eval()`, dropout disabled (`dropout_rate=0.0`)
- Warmup: `10` iterations
- Measurement: `50` iterations, average seconds per batch
- Python: `C:/Users/mukil/AppData/Local/Programs/Python/Python310/python.exe`
- Torch: `2.6.0+cu126`

## Throughput Results

| Device | Backbone | Sec/Batch | Img/Sec | Relative |
|---|---|---:|---:|---:|
| CPU | `lprnet` | 0.098307 | 325.51 | baseline |
| CPU | `svtr_lcnet` | 0.070830 | 451.79 | +38.8% |
| CUDA | `lprnet` | 0.009113 | 3511.65 | baseline |
| CUDA | `svtr_lcnet` | 0.007872 | 4065.08 | +15.8% |

## Output Contract Check

Both backbones preserve the recognizer output contract:

- Output rank is unchanged: `(N, class_num, T)`
- CTC interface remains unchanged in training/testing scripts

Observed output widths:

- `lprnet`: `T=74`
- `svtr_lcnet`: `T=47`

This is acceptable because `T_length` is computed dynamically in training and decoding still operates on the same tensor contract.

## Notes

- These are synthetic throughput numbers, not accuracy metrics.
- Accuracy comparison requires training both backbones on the same data split and evaluating with identical decode settings.
- If needed, we can further tune hybrid speed/quality using:
  - `--mix_blocks` (2 -> 1 for lower cost)
  - `--mix_dim` (192 -> 160/128 for lower cost)
  - `--mix_heads` (keep divisor of `mix_dim`)
