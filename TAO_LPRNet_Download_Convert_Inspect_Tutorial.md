# TAO LPRNet: Download, Convert (`.onnx` / `.engine`), and Visualize Layers

This tutorial gives a practical end-to-end workflow for NVIDIA TAO LPRNet models:

1. Download a pretrained TAO LPRNet model from NGC.
2. Convert/export it to `.onnx` (if supported by your TAO version) and/or `.engine`.
3. Print and visualize layer details.
4. Save layer information to files.

---

## 0) Important Notes Before You Start

- TAO workflows are Linux/container-first. On Windows, use **WSL2 + Ubuntu** (recommended) or a Linux machine.
- LPRNet export behavior differs between TAO versions:
  - Some versions export primarily to `.etlt`.
  - Newer TAO deploy flows also support `.onnx` as model input for engine generation.
- `.engine` files are **GPU and TensorRT version specific**.

---

## 1) Prerequisites

Install and verify:

- NVIDIA GPU driver
- Docker + NVIDIA Container Toolkit
- TAO Toolkit (`tao` CLI)
- NGC CLI (`ngc`)
- Optional tools:
  - `trtexec` (for TensorRT layer dump)
  - `polygraphy` (for ONNX inspection)
  - Netron (GUI model graph viewer)

Quick checks:

```bash
nvidia-smi
docker --version
tao --help
ngc --version
```

---

## 2) Create a Working Directory Structure

```bash
mkdir -p ~/tao_lprnet/{models,specs,export,layers}
cd ~/tao_lprnet
```

Folders:

- `models/` -> downloaded pretrained files
- `specs/` -> TAO experiment spec(s)
- `export/` -> `.etlt`, `.onnx`, `.engine`
- `layers/` -> saved layer dumps

---

## 3) Login to NGC and Download LPRNet

### 3.1 Configure NGC CLI

```bash
ngc config set
```

Provide your API key when prompted.

### 3.2 Discover the exact LPRNet model URI/version

```bash
ngc registry model list nvidia/tao/lprnet
```

If needed, inspect versions:

```bash
ngc registry model info nvidia/tao/lprnet:<VERSION>
```

### 3.3 Download a chosen version

```bash
ngc registry model download-version nvidia/tao/lprnet:<VERSION> --dest ~/tao_lprnet/models
```

After download, check what you got:

```bash
ls -R ~/tao_lprnet/models
```

You will usually see model artifacts such as `.tlt` or `.etlt` plus metadata/readme files.

---

## 4) Prepare a Spec File for Export/Deploy

Create `~/tao_lprnet/specs/lprnet_export_spec.txt`:

```proto
random_seed: 42
lpr_config {
  hidden_units: 512
  max_label_length: 8
  arch: "baseline"
  nlayers: 10
}
augmentation_config {
  output_width: 96
  output_height: 48
  output_channel: 3
}
eval_config {
  batch_size: 1
}
```

Use your training spec if you already have one. Keep input geometry consistent with the model.

---

## 5) Export/Convert Model Artifacts

Set reusable variables first:

```bash
export KEY="<YOUR_TAO_KEY>"
export MODEL_IN="<PATH_TO_DOWNLOADED_MODEL>"     # .tlt or .etlt
export SPEC=~/tao_lprnet/specs/lprnet_export_spec.txt
export OUT=~/tao_lprnet/export
mkdir -p "$OUT"
```

### 5A) Export to `.etlt` (works broadly across TAO versions)

```bash
tao model lprnet export \
  -m "$MODEL_IN" \
  -k "$KEY" \
  -e "$SPEC" \
  -o "$OUT/lprnet.etlt"
```

### 5B) Export to `.onnx` (version-dependent)

Because CLI options vary, check help first:

```bash
tao model lprnet export --help
```

Then use the ONNX option shown by your version (commonly an ONNX output flag or ONNX output path).

Example pattern:

```bash
# Replace with the exact ONNX export option shown in --help
tao model lprnet export \
  -m "$MODEL_IN" \
  -k "$KEY" \
  -e "$SPEC" \
  <ONNX_EXPORT_OPTION> "$OUT/lprnet.onnx"
```

If your TAO version does not support ONNX export for LPRNet in this command, use the `.etlt -> .engine` path directly (next step).

### 5C) Generate TensorRT engine (`.engine`)

#### Option 1: TAO Deploy (accepts `.onnx` or `.etlt` depending on version/docs)

```bash
tao deploy lprnet gen_trt_engine \
  -m "$OUT/lprnet.onnx" \
  -e "$SPEC" \
  -r "$OUT" \
  --data_type fp16 \
  --max_batch_size 16 \
  --engine_file "$OUT/lprnet_fp16.engine"
```

If you only have `.etlt`, set `-m "$OUT/lprnet.etlt"` and add `-k "$KEY"` if required.

#### Option 2: tao-converter (classic `.etlt -> .engine`)

```bash
tao-converter "$OUT/lprnet.etlt" \
  -k "$KEY" \
  -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
  -e "$OUT/lprnet_fp16.engine"
```

---

## 6) Print and Save Layer Info

## 6.1 ONNX layers (CLI)

Install Polygraphy if needed:

```bash
python3 -m pip install -U polygraphy onnx
```

Print layers:

```bash
polygraphy inspect model "$OUT/lprnet.onnx" --show layers
```

Save to file:

```bash
polygraphy inspect model "$OUT/lprnet.onnx" --show layers > ~/tao_lprnet/layers/lprnet_onnx_layers.txt
```

### 6.2 ONNX layers (Python script)

Create `~/tao_lprnet/layers/print_onnx_layers.py`:

```python
import onnx

model_path = "/home/$USER/tao_lprnet/export/lprnet.onnx"
out_path = "/home/$USER/tao_lprnet/layers/lprnet_onnx_layers_python.txt"

model = onnx.load(model_path)
lines = []
for i, node in enumerate(model.graph.node):
    lines.append(f"{i:04d} | op={node.op_type} | name={node.name}")
    if node.input:
        lines.append(f"      in : {list(node.input)}")
    if node.output:
        lines.append(f"      out: {list(node.output)}")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Saved: {out_path}")
```

Run:

```bash
python3 ~/tao_lprnet/layers/print_onnx_layers.py
```

### 6.3 TensorRT engine layers (`trtexec`)

```bash
trtexec \
  --loadEngine="$OUT/lprnet_fp16.engine" \
  --dumpLayerInfo \
  --profilingVerbosity=detailed
```

Save output:

```bash
trtexec \
  --loadEngine="$OUT/lprnet_fp16.engine" \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  > ~/tao_lprnet/layers/lprnet_engine_layers.txt 2>&1
```

---

## 7) Visualize the Model Graph

## 7.1 Netron (recommended for ONNX)

```bash
python3 -m pip install -U netron
netron "$OUT/lprnet.onnx" --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` in your browser.

If you installed desktop Netron, just open `lprnet.onnx` from the app.

### 7.2 TensorRT graph visualization note

- `.engine` is compiled/optimized, so visualization is less straightforward than ONNX.
- For readable architecture inspection, ONNX + Netron is usually best.

---

## 8) Quick End-to-End Command Sequence (Template)

```bash
# 1) discover + download
ngc config set
ngc registry model list nvidia/tao/lprnet
ngc registry model download-version nvidia/tao/lprnet:<VERSION> --dest ~/tao_lprnet/models

# 2) export / convert
tao model lprnet export -m <MODEL_IN> -k <KEY> -e ~/tao_lprnet/specs/lprnet_export_spec.txt -o ~/tao_lprnet/export/lprnet.etlt
tao deploy lprnet gen_trt_engine -m ~/tao_lprnet/export/lprnet.etlt -k <KEY> -e ~/tao_lprnet/specs/lprnet_export_spec.txt -r ~/tao_lprnet/export --data_type fp16 --engine_file ~/tao_lprnet/export/lprnet_fp16.engine

# 3) inspect layers
trtexec --loadEngine=~/tao_lprnet/export/lprnet_fp16.engine --dumpLayerInfo --profilingVerbosity=detailed > ~/tao_lprnet/layers/lprnet_engine_layers.txt 2>&1
```

---

## 9) Troubleshooting

- `tao: command not found`
  - Activate TAO launcher/container environment and retry.
- `model decrypt/export failed`
  - The TAO key is wrong. Confirm `-k <KEY>`.
- `ONNX export option not recognized`
  - Your TAO version differs. Use `tao model lprnet export --help` and follow shown flags; fallback to `.etlt -> .engine`.
- `trtexec: command not found`
  - Install TensorRT tools or use the TAO deploy container with TensorRT utilities.
- Engine fails on another machine
  - Rebuild `.engine` on target GPU/TensorRT stack.

---

## 10) What to Share/Archive

For reproducibility, keep these files together:

- `specs/lprnet_export_spec.txt`
- `export/lprnet.etlt` and/or `export/lprnet.onnx`
- `export/lprnet_fp16.engine`
- `layers/lprnet_onnx_layers.txt`
- `layers/lprnet_engine_layers.txt`

This gives you both deployable artifacts and human-readable layer records.
