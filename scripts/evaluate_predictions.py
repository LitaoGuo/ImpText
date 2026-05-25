#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imptext.evaluation import Evaluator


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


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
    parser = argparse.ArgumentParser(description="Recompute ImpText metrics from a results JSONL file.")
    parser.add_argument("--dataset", default="imptext_bench/dataset.jsonl")
    parser.add_argument("--results", required=True, help="JSONL with id and prediction_raw fields.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    results = {str(r["id"]): r.get("prediction_raw", r.get("prediction", "")) for r in load_jsonl(Path(args.results))}

    predictions = []
    ground_truths = []
    for sample in dataset:
        sample_id = str(sample.get("id"))
        predictions.append(results.get(sample_id, ""))
        ground_truths.append(build_ground_truth(sample))

    metrics = Evaluator(match_threshold=args.threshold).evaluate(predictions, ground_truths)
    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
