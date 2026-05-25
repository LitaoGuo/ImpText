# ImpText-Bench Metadata

This directory contains the benchmark JSONL manifests. The image files are not
stored in the GitHub repository; they will be released separately.

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

Files tracked in GitHub:

- `dataset.jsonl`: full 1,630-record manifest.
- `dataset_available.jsonl`: image-backed manifest for the external data
  package; equivalent to `dataset.jsonl` once all benchmark images are
  downloaded.
- `missing_images.txt`: expected to be empty after the external image package is
  placed under `images/`.

Dataset image assets will be released separately.

After downloading the image package, validate local coverage from the repository
root:

```bash
python scripts/check_benchmark.py \
  --dataset imptext_bench/dataset.jsonl \
  --image-root imptext_bench/images
```
