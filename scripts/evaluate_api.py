#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imptext.engines import APIEngine
from imptext.evaluation import Evaluator


PROMPT = """Analyze the image to determine if it contains any hidden text, which may include contact details, URLs, or other concealed messages.
Your final output must indicate "Yes" or "No" and provide the content of the hidden text found.
Please strictly adhere to the following output format:
<has_hidden_text> [Yes/No] </has_hidden_text> <hidden_content> [The extracted text or "None"] </hidden_content>"""


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def resolve_image_path(image_root: Path, image_path: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str(image_root / p)


def meta_category(sample: dict) -> str:
    if sample.get("is_white_sample", False):
        return "BENIGN SAMPLES"
    category = str(sample.get("category", "")).lower()
    if "physical" in category or "deformation" in category or "stretch" in category:
        return "PHYSICAL DEFORMATION"
    if "dialogue" in category or "completion" in category or "cognitive" in category:
        return "COGNITIVE SUGGESTION"
    return "VISUAL CAMOUFLAGE"


def build_ground_truth(sample: dict) -> dict:
    return {
        "has_hidden_text": not bool(sample.get("is_white_sample", False)),
        "hidden_content": sample.get("hidden_content", ""),
        "category": meta_category(sample),
        "id": sample.get("id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an OpenAI-compatible multimodal model on ImpText-Bench.")
    parser.add_argument("--model", required=True, help="Model name sent to the API.")
    parser.add_argument("--dataset", default="imptext_bench/dataset.jsonl", help="Path to ImpText-Bench JSONL manifest.")
    parser.add_argument("--image-root", default="imptext_bench/images", help="Root directory containing white/ and black/ images.")
    parser.add_argument("--output-dir", default="outputs/api_eval", help="Directory for results and metrics.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5, help="NED tolerance threshold tau for TMS. Default: 0.5.")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke tests.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip records whose image file is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and exit before initializing the API client.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    image_root = Path(args.image_root)
    output_root = Path(args.output_dir)
    model_dir = output_root / safe_name(args.model)
    results_path = model_dir / "results.jsonl"
    metrics_path = model_dir / "metrics.json"

    dataset = load_jsonl(dataset_path)
    if args.limit > 0:
        dataset = dataset[: args.limit]

    processed = set()
    if results_path.exists():
        for record in load_jsonl(results_path):
            if "id" in record:
                processed.add(str(record["id"]))

    items = []
    missing = []
    for sample in dataset:
        sample_id = str(sample.get("id", len(items)))
        image_path = resolve_image_path(image_root, sample["image_path"])
        if not os.path.exists(image_path):
            missing.append({"id": sample_id, "image_path": image_path})
            if args.skip_missing:
                continue
        if sample_id in processed:
            continue
        items.append({"id": sample_id, "image_path": image_path, "prompt": PROMPT, "sample": sample})

    if missing and not args.skip_missing:
        print(f"Missing {len(missing)} images. First missing: {missing[0]['image_path']}", file=sys.stderr)
        print("Use --skip-missing for metadata-only smoke tests, or place images under --image-root.", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dataset records after limit: {len(dataset)}")
        print(f"Images missing: {len(missing)}")
        print(f"Already processed: {len(processed)}")
        print(f"Samples that would be sent to the API: {len(items)}")
        print("Dry run complete. No API client was initialized and no requests were sent.")
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)
    engine = APIEngine(model=args.model, concurrency=args.concurrency)
    evaluator = Evaluator(match_threshold=args.threshold)

    with results_path.open("a", encoding="utf-8") as f:
        for item, result in engine.generate(items):
            sample = item["sample"]
            gt = build_ground_truth(sample)
            parsed = evaluator.parse_output(result["text"])
            similarity = None
            normalized_edit_distance = None
            pass_threshold = None
            if gt["has_hidden_text"]:
                similarity = evaluator.compute_text_similarity(parsed["hidden_content"], gt["hidden_content"])
                normalized_edit_distance = evaluator.compute_normalized_edit_distance(
                    parsed["hidden_content"], gt["hidden_content"]
                )
                pass_threshold = normalized_edit_distance <= args.threshold
            record = {
                "id": item["id"],
                "category": gt["category"],
                "image_path": sample["image_path"],
                "ground_truth": gt,
                "prediction_raw": result["text"],
                "prediction_parsed": parsed,
                "similarity": similarity,
                "normalized_edit_distance": normalized_edit_distance,
                "pass_threshold": pass_threshold,
                "request_meta": {
                    "success": result["success"],
                    "attempts": result["attempts"],
                    "last_error": result["last_error"],
                },
                "token_usage": result["usage"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    all_results = {str(r["id"]): r.get("prediction_raw", "") for r in load_jsonl(results_path) if "id" in r}
    predictions = []
    ground_truths = []
    for sample in dataset:
        sample_id = str(sample.get("id"))
        if args.skip_missing:
            image_path = resolve_image_path(image_root, sample["image_path"])
            if not os.path.exists(image_path):
                continue
        predictions.append(all_results.get(sample_id, ""))
        ground_truths.append(build_ground_truth(sample))

    metrics = evaluator.evaluate(predictions, ground_truths)
    metrics["model"] = args.model
    metrics["timestamp"] = dt.datetime.now().isoformat()
    metrics["num_predictions"] = len(predictions)
    metrics["missing_images"] = len(missing)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Results: {results_path}")
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
