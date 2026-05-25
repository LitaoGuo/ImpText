# ImpText

**ImpText-Bench is a benchmark and evaluation toolkit for implicit text reasoning in multimodal models.**

ImpText focuses on images where target text is hidden, distorted, visually camouflaged, or recoverable only through context. This repository contains benchmark metadata, evaluation scripts, OCR baselines, threshold analysis, image-enhancement tools, and README figure assets.

Benchmark images are not stored in this GitHub repository. They will be released separately and should be placed locally using the paths below after download.

![ImpText-Bench taxonomy and representative examples](docs/assets/figures/showbench.png)

## Benchmark

ImpText-Bench contains **1,630** image-text records:

| Category | Count |
| --- | ---: |
| Benign Samples | 1,141 |
| Physical Deformation | 58 |
| Adversarial Texture | 134 |
| Visual Distraction | 93 |
| Environmental Fusion | 73 |
| Irregular Typography | 28 |
| Contextual Completion | 61 |
| Implicit Dialogue | 42 |
| **Total** | **1,630** |

The manifest lives in [imptext_bench/dataset.jsonl](imptext_bench/dataset.jsonl). Images are expected under `imptext_bench/images/` after downloading the external data package. Each line is a JSON object:

```json
{
  "id": "1142",
  "category": "Adversarial texture",
  "image_path": "black/1142.png",
  "is_white_sample": false,
  "hidden_content": "..."
}
```

Expected image layout:

```text
imptext_bench/images/
  white/<id>.png
  black/<id>.png
```

Dataset image assets will be released separately.

After downloading the images into `imptext_bench/images/`, check completeness:

```bash
python scripts/check_benchmark.py \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images
```

Expected summary:

```text
records: 1630
available_images: 1630
missing_images: 0
black: 489
white: 1141
```

## Framework Figure

![ImpText-Reader training framework](docs/assets/figures/pipeline_imptext3.png)

PDF source: [pipeline_imptext3.pdf](docs/assets/figures/pipeline_imptext3.pdf).

This repository provides benchmark metadata and evaluation utilities. Model training code and model weights are outside the scope of this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py
```

## Evaluation

Metrics include hidden-text classification and Text Match Score (TMS), based on normalized edit distance with default threshold `0.5`. Expected model output:

```xml
<has_hidden_text> Yes/No </has_hidden_text>
<hidden_content> recovered text or None </hidden_content>
```

```bash
export API_KEY="..."
export API_BASE_URL="https://your-openai-compatible-endpoint/v1"

python scripts/evaluate_api.py \
  --model your-model \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images \
  --output-dir outputs/api_eval \
  --concurrency 4
```

Dry-run without requests:

```bash
python scripts/evaluate_api.py \
  --model dummy \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images \
  --skip-missing \
  --limit 3 \
  --dry-run
```

Recompute metrics:

```bash
python scripts/evaluate_predictions.py \
  --dataset imptext_bench/dataset.jsonl \
  --results outputs/api_eval/your-model/results.jsonl \
  --threshold 0.5 \
  --output outputs/api_eval/your-model/metrics_recomputed.json
```

## OCR Baselines

OCR-style multimodal prompt baseline:

```bash
python scripts/evaluate_ocr.py \
  --engine openai \
  --model your-model \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images \
  --output-dir outputs/ocr_eval
```

Generic HTTP JSON OCR service:

```bash
export OCR_API_KEY="..."

python scripts/evaluate_ocr.py \
  --engine http-json \
  --model paddleocr-compatible \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images \
  --http-url "https://your-ocr-endpoint.example/ocr" \
  --api-key-env OCR_API_KEY \
  --api-key-header Authorization \
  --api-key-prefix Bearer \
  --response-field "result.ocrResults.*.prunedResult" \
  --output-dir outputs/ocr_eval
```

No private OCR endpoint or token is included.

## Threshold Sweep

```bash
python scripts/tau_sweep.py \
  --dataset imptext_bench/dataset.jsonl \
  --results ModelA=outputs/api_eval/model-a/results.jsonl \
  --results OCR=outputs/ocr_eval/ocr/results.jsonl \
  --thresholds 0.1,0.3,0.5,0.7,0.9 \
  --output-dir outputs/tau_sweep
```

Outputs: `tau_sweep.json`, `tau_sweep.csv`, and `tau_sweep.md`.

## Image Tools

```bash
python scripts/apply_tool.py \
  --image imptext_bench/images/black/1142.png \
  --tool clahe \
  --output outputs/tools/1142_clahe.png
```

Tools include `adaptive_thresholding`, `canny_edge_extraction`, `channel_extraction_*`, `jpeg_purify`, `posterization`, `sharpening`, `anisotropic_stretch`, `clahe`, `downscale_*`, `morphological_closing`, and `blackhat_extraction`.

## Tests

```bash
python -m unittest discover -s tests
python scripts/evaluate_api.py --model dummy --limit 3 --skip-missing --dry-run
python scripts/evaluate_ocr.py --engine openai --model dummy --limit 3 --skip-missing --dry-run
```
