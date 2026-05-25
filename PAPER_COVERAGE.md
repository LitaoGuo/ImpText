# Paper Coverage Matrix

This document compares the public `release-version/` package against the local
development `Code/` directory and the claims in `imptext.tex`.

## Covered In This Release

| Paper component | Status | Release files |
| --- | --- | --- |
| ImpText-Bench manifest | Covered | `imptext_bench/dataset.jsonl` |
| ImpText-Bench images | Covered | `imptext_bench/images/white`, `imptext_bench/images/black` |
| Dataset taxonomy and counts | Covered | `imptext_bench/README.md`, `scripts/check_benchmark.py` |
| F1 / Recall / Accuracy | Covered | `src/imptext/evaluation/evaluator.py` |
| TMS with NED threshold | Covered | `src/imptext/evaluation/evaluator.py` |
| OpenAI-compatible MLLM evaluation | Covered | `scripts/evaluate_api.py`, `src/imptext/engines/api_engine.py` |
| Saved-prediction metric recomputation | Covered | `scripts/evaluate_predictions.py` |
| Visual Enhancer tool library | Covered | `src/imptext/tools/image_robustness_toolkit.py`, `src/imptext/tools/wrapper.py` |
| Release smoke tests | Covered | `scripts/check_setup.py`, `tests/` |

## Missing For Full Method Reproduction

| Paper component | Current status | Related local `Code/` files | Why not included directly |
| --- | --- | --- | --- |
| ImpText-Reader trained inference pipeline | Missing | `Code/model/scripts/run_test_multiturn.py`, `Code/model/scripts/run_test_multiturn_full_context.py` | Depends on local model paths and training outputs. Needs a sanitized CLI and released checkpoints. |
| Visual Enhancer learned tool selector | Missing | `Code/rebuttal/src/run_test_api_react_baseline.py`, `Code/rebuttal/src/run_test_api_best_of_12.py`, `Code/rebuttal/src/tools_wrapper.py` | Release includes tool execution, but not a trained selector policy or oracle-selection dataset. |
| Phase I Knowledge Warm-up SFT | Missing | `Code/model/scripts/run_sft.sh`, `Code/model/scripts/prepare_sft_data.py` | Hard-coded internal cluster paths and LLaMA-Factory environment. |
| Phase II Oracle-Guided Co-Training | Missing | `Code/rebuttal/src/run_test_api_best_of_12.py`, `Code/model/scripts/process_sft_datasets.py` | Needs a cleaned oracle-search/pass-rate pipeline and public intermediate data. |
| Boundary-aware data filtering | Missing | Scattered in model/rebuttal scripts and outputs | No standalone sanitized script currently exists in release form. |
| Phase III Boundary-Aware RL / GRPO | Missing | `Code/model/data_processed_verl/*.parquet`, `Code/model/scripts/*`, external verl setup | Requires a separate sanitized training release with verl configs and model artifacts. |
| Final ImpText-Reader checkpoint / LoRA | Missing | Local training outputs only | No public checkpoint is included. |
| ReAct + Tool baseline | Missing | `Code/rebuttal/src/run_test_api_react_baseline.py` | Valuable but has local paths/output assumptions; should be cleaned before release. |
| Best-of-tool baseline | Missing | `Code/rebuttal/src/run_test_api_best_of_12.py` | Valuable but has local paths/output assumptions; should be cleaned before release. |
| OCR baselines | Covered with generic adapters | `Code/test_api/src/main_ocr.py`, `Code/test_api/src/engines/paddle_engine.py` | Release provides `scripts/evaluate_ocr.py` without hard-coded tokens or private service URLs. |
| Threshold/tau sweep tables | Covered | `Code/rebuttal/tao_selectiom_ablation/analyze_tau_sensitivity.py` | Release provides `scripts/tau_sweep.py` for JSON/CSV/Markdown threshold sweeps. |
| Semantic threshold calibration with judge model | Missing | `Code/rebuttal/tao_selectiom_ablation/analyze_judge_threshold_calibration.py` | Depends on judge-model outputs and API traces. |
| Data regeneration pipeline | Missing | `Code/src/pipeline*.py`, `Code/src/seedream4_api.py`, `Code/src/prompt_generator.py` | Depends on private raw annotations and Seedream/Doubao APIs. Should be a separate appendix if released. |

## Recommended Next Additions

1. Add cleaned generic tool-use baselines:
   - `react_tool_baseline.py`.
   - `best_of_tool_baseline.py`.
2. If method reproducibility is required, create a separate training release:
   - SFT data conversion.
   - Oracle search.
   - Boundary filtering.
   - GRPO config.
   - Checkpoint or LoRA links.

## Release Positioning

The current package should be described as:

> ImpText-Bench public release with image data, evaluation metrics, API
> evaluation utilities, and the image-enhancement tool library.

It should not yet be described as a complete reproduction package for all
ImpText-Reader training stages.
