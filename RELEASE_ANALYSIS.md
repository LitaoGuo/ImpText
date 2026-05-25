# Release Analysis

This file records what was selected from the local `Code/` directory and what
should stay out of the GitHub release.

## Paper Alignment

`imptext.tex` describes three public-facing assets:

1. **ImpText-Bench**: 1,630 image-text records, including benign samples and
   implicit-text categories.
2. **Evaluation protocol**: binary existence classification plus Text Match
   Score, implemented with normalized Levenshtein similarity and threshold 0.5.
3. **ImpText-Reader/tool augmentation**: a visual enhancement library and a
   tool-augmented recognition workflow.

The release version therefore keeps the benchmark manifest, evaluation metrics,
API evaluation runner, and image enhancement tools.

The two provided image archives were also normalized into the manifest-compatible
layout. They cover all black/implicit samples but only part of the benign split.

## Included

- `Code/imptext_bench/dataset.jsonl` -> `imptext_bench/dataset.jsonl`
  - Safe relative paths.
  - 1,630 records.
  - Needed for all evaluation.
- `imptext-1.zip` -> `imptext_bench/images/white/*.png`
  - 571 benign images, IDs `0001` through `0571`.
- `imptext-3.zip` -> `imptext_bench/images/white/*.png`
  - 570 benign images, IDs `0572` through `1141`.
- `imptext-2.zip` -> `imptext_bench/images/black/*.png`
  - 489 implicit images, IDs `1142` through `1630`.
- Generated release manifests:
  - `imptext_bench/dataset_available.jsonl`: 1,630 image-backed records.
  - `imptext_bench/missing_images.txt`: empty; no missing manifest images.
- `Code/test_api/src/metrics/evaluator.py` -> `src/imptext/evaluation/evaluator.py`
  - Core F1 and TMS logic.
  - Directly matches the paper's metric definition.
- `Code/test_api/src/utils/image_robustness_toolkit.py` -> `src/imptext/tools/image_robustness_toolkit.py`
  - Public implementation of the visual enhancement tool library.
- New cleaned scripts:
  - `scripts/check_setup.py`
  - `scripts/check_benchmark.py`
  - `scripts/evaluate_ocr.py`
  - `scripts/evaluate_api.py`
  - `scripts/evaluate_predictions.py`
  - `scripts/tau_sweep.py`
  - `scripts/apply_tool.py`
- Basic tests:
  - `tests/test_evaluator.py`
  - `tests/test_manifest.py`

## Excluded

- `.venv/`, `__pycache__/`, `.DS_Store`
  - Generated local files.
- `outputs_api_eval*`, `outputs_api_ocr`, `rebuttal/outputs_*`, `model/result_*`
  - Historical experiment outputs with local paths and large JSONL traces.
- Shell scripts such as `run_v1.sh`, `run_v2.sh`, `run_v3.sh`, `run_ocr.sh`
  - Contain real API keys and provider-specific endpoints.
- `Code/.trae/`
  - Private development notes and debugging records.
- `Code/model/data_processed_verl/*.parquet`
  - Training data artifacts, not suitable for a lightweight GitHub code release.
- `Code/model/scripts/*` as-is
  - Many scripts hard-code internal cluster paths, endpoint IDs, and local
    training infra.
- `Code/rebuttal/src/config.py` as-is
  - Contains a fallback Doubao key and internal paths.
- `Code/imptext_bench/imptext-1.zip` as a committed zip
  - About 537 MB and not git-friendly.
  - It was extracted and normalized instead.
- Raw zip archives in general
  - Prefer Git LFS or external release artifacts.

## Recommended Public Repo Shape

```text
ImpText/
  README.md
  requirements.txt
  .gitignore
  imptext_bench/
    README.md
    dataset.jsonl
    dataset_available.jsonl
    missing_images.txt
    images/
  src/imptext/
    engines/
    evaluation/
    tools/
  scripts/
    check_benchmark.py
    evaluate_api.py
    evaluate_predictions.py
    apply_tool.py
  tests/
```

## Follow-up Before Publishing

1. Attach the full image archive outside git if not using Git LFS, for example GitHub Releases or Hugging Face
   Datasets.
2. Add citation, license, and dataset use policy files.
3. If releasing training code, create a separate sanitized training appendix
   instead of copying `Code/model` directly.
