# ImpText-Bench Metadata

This directory contains the benchmark JSONL manifests. The image files are
hosted outside the GitHub code repository.

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

Files tracked in GitHub:

- `dataset.jsonl`: full 1,630-record manifest.
- `dataset_available.jsonl`: image-backed manifest for the external data
  package; equivalent to `dataset.jsonl` once all benchmark images are
  downloaded.
- `missing_images.txt`: expected to be empty after the external image package is
  placed under `images/`.

Dataset image assets are distributed as an external package.

After downloading the image package, validate local coverage from the repository
root:

```bash
python scripts/check_benchmark.py \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images
```
