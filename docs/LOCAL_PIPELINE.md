# Local Pipeline Guide (No AWS)

Run the P&ID digitization pipeline on your machine as a **standalone Python CLI script**. This is **not a web app** — there is no server, browser UI, or REST API.

**No API keys or AWS credentials are required.** Everything runs locally after you install dependencies and download the pre-trained model once.

---

## What it does

`local_pipeline.py` chains the same processing logic as the AWS pipeline:

| Step | Module | Technology |
|------|--------|------------|
| 1 | Notes preprocessing | OpenCV / PIL |
| 2 | Text detection | PaddleOCR (replaces Bedrock) |
| 3 | Symbol detection | PyTorch (replaces SageMaker) |
| 4 | Line detection | OpenCV |
| 5 | Graph generation | NetworkX + DEXPI export |
| 6 | Visualization (optional) | Matplotlib |

---

## Prerequisites

- **Python 3.12+** (3.13 also works)
- **NVIDIA GPU + CUDA** recommended (CPU works but is slow for symbol detection)
- A **P&ID image** in PNG or JPG format
- **Disk space**: ~2 GB for Python packages + model weights

---

## Step 1 — Get the code

```bash
git clone <your-repo-url>
cd guidance-for-pind-digitization-on-aws
```

If the local pipeline is on a feature branch:

```bash
git checkout cursor/local-pipeline-paddleocr-7419
```

---

## Step 2 — Create a virtual environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## Step 3 — Install dependencies

Install PyTorch with CUDA first (adjust the CUDA version if needed):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then install the project requirements:

```bash
pip install -r inference/requirements.txt
pip install -r local/requirements.txt
```

**Verify PyTorch sees your GPU (optional):**

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Expected output: `CUDA: True`

---

## Step 4 — Download the pre-trained model

The symbol detection model is hosted on GitHub Releases (public download, no login):

```bash
wget https://github.com/aws-solutions-library-samples/guidance-for-piping-and-instrumentation-diagrams-digitization-on-aws/releases/download/v1.0.0/model.tar.gz

mkdir -p models
tar -xzf model.tar.gz -C models
```

**Verify the model files exist:**

```bash
ls models/
# Expected: frcnn_checkpoint_50000.pth  last-v9.ckpt  references/  (and possibly other files)
```

---

## Step 5 — Run the pipeline

Replace `/path/to/your-pnid.png` with your diagram file.

**Basic run:**

```bash
python local_pipeline.py /path/to/your-pnid.png \
  --model-dir ./models \
  --output ./output
```

**With visualization PNGs:**

```bash
python local_pipeline.py /path/to/your-pnid.png \
  --model-dir ./models \
  --output ./output \
  --visualize
```

**Verbose logging:**

```bash
python local_pipeline.py /path/to/your-pnid.png \
  --model-dir ./models \
  --output ./output \
  --visualize \
  --verbose
```

---

## Step 6 — Check the output

Results are written under `./output/<execution-id>/`. Example:

```
output/local-20250630-120000/
├── execution_summary.json
├── config/
│   └── processing_config.json
├── input/
│   └── your-pnid.png
├── notes-processing/
│   └── processed_image.png
├── text-detection/
│   ├── text_detection_results.json
│   └── debug_image.png
├── symbol-detection/
│   └── detections.json
├── line-detection/
│   ├── detected_lines.json
│   └── debug_image.png
├── graph/
│   ├── graph_data.json
│   └── dexpi_output.xml
└── visualization/              # only when --visualize is used
    ├── physical_layout.png
    └── graph_representation.png
```

Open `execution_summary.json` for a quick overview of counts and file paths.

---

## Command-line options

| Flag | Description |
|------|-------------|
| `--output`, `-o` | Output directory (default: `./output`) |
| `--model-dir` | Folder with extracted `model.tar.gz` (default: `./models`) |
| `--config` | Custom JSON config (default: `cdk/lambda/input_validator/default_config.json`) |
| `--visualize` | Generate physical layout and graph representation PNGs |
| `--skip-notes` | Disable automatic frame/notes removal |
| `--cpu` | Force CPU for PaddleOCR (symbol detection still uses GPU if available) |
| `--execution-id` | Custom folder name for this run |
| `--verbose`, `-v` | Debug logging |

---

## Custom configuration (optional)

Copy and edit the default config:

```bash
cp cdk/lambda/input_validator/default_config.json my_config.json
# edit my_config.json
python local_pipeline.py diagram.png --model-dir ./models --config my_config.json
```

See [Configuration Guide](CONFIGURATION.md) for parameter details.

---

## First-run notes

- **PaddleOCR** downloads OCR model weights automatically on first run (~100 MB). No API key needed.
- **Symbol detection** loads PyTorch checkpoints from `./models` — the first inference may take a minute.
- **OCR quality** may differ from the AWS Bedrock pipeline on skewed or dense engineering drawings.

---

## Troubleshooting

### `Model directory not found` or missing `.pth` / `.ckpt`

Re-run Step 4 and confirm files are inside `./models`, not a nested subfolder.

### `ModuleNotFoundError` (e.g. `networkx`, `paddleocr`, `torch`)

Activate your virtual environment and re-run Step 3.

### `CUDA: False` but you have a GPU

Reinstall PyTorch with the CUDA build matching your driver:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Out of memory on symbol detection

Use a smaller image, or run on a machine with more VRAM. Symbol detection uses Faster R-CNN + Siamese networks on GPU.

### PaddleOCR errors on macOS

`local/requirements.txt` installs CPU-only PaddlePaddle on macOS. Use `--cpu` if needed.

---

## Local vs AWS pipeline

| | Local pipeline | AWS pipeline |
|--|----------------|--------------|
| Entry point | `python local_pipeline.py` | `cdk deploy` + Step Functions |
| Text OCR | PaddleOCR | Bedrock Data Automation |
| Symbol detection | Local PyTorch | SageMaker endpoint |
| Storage | Local filesystem | S3 |
| API keys | **None** | AWS credentials |
| Cost | Your hardware only | AWS service charges |

For AWS deployment, see [cdk/README.md](../cdk/README.md) and the main [README](../README.md).
