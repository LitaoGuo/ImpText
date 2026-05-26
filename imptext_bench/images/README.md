# Image Directory

This directory is the local target for ImpText-Bench image files downloaded
from the Hugging Face dataset
[`Riversideli/ImpText-Bench`](https://huggingface.co/datasets/Riversideli/ImpText-Bench).

From the code repository root, the Hugging Face download command should target
`imptext_bench` so that the dataset's `images/` directory lands here:

```bash
hf download Riversideli/ImpText-Bench \
  --repo-type dataset \
  --include "images/**" \
  --local-dir imptext_bench
```

Expected local layout:

```text
imptext_bench/images/
  white/<id>.png
  black/<id>.png
```

Expected complete local coverage:

- `white/`: 1,141 benign images
- `black/`: 489 implicit-text images
