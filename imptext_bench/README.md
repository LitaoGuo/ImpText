# ImpText-Bench Metadata

This directory contains the benchmark JSONL manifests. The image files are
distributed through the Hugging Face dataset
[`Riversideli/ImpText-Bench`](https://huggingface.co/datasets/Riversideli/ImpText-Bench).

Expected image layout:

```text
images/
  white/<id>.png
  black/<id>.png
```

The full manifest has 1,630 records:

- 1,141 benign samples.
- 489 implicit-text samples across physical deformation, visual camouflage, and
  cognitive suggestion categories.

The taxonomy follows the ImpText-Bench definition:

| Primary category | Sub-category | Count |
| --- | --- | ---: |
| Benign Samples | Normal Images | 1,141 |
| Physical Deformation | Stretching | 58 |
| Visual Camouflage | Adversarial Texture | 134 |
| Visual Camouflage | Visual Distraction | 93 |
| Visual Camouflage | Environmental Fusion | 73 |
| Visual Camouflage | Irregular Typography | 28 |
| Cognitive Suggestion | Contextual Completion | 61 |
| Cognitive Suggestion | Implicit Dialogue | 42 |
| **Total** | **All Samples** | **1,630** |

Files in this directory:

- `dataset.jsonl`: full 1,630-record manifest.

Dataset image assets are distributed through the Hugging Face dataset.

After downloading the image files into the ImpText code repository, validate
local coverage from the code repository root:

```bash
python scripts/check_benchmark.py \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images
```

If needed, the same script can write local diagnostic manifests:

```bash
python scripts/check_benchmark.py \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images \
  --write-available outputs/available_records.jsonl \
  --write-missing outputs/missing_image_paths.txt
```
