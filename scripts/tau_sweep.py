#!/usr/bin/env python3
import argparse
import csv
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


def build_ground_truths(dataset_path: Path) -> tuple[list[str], list[dict]]:
    ids = []
    ground_truths = []
    for sample in load_jsonl(dataset_path):
        sample_id = str(sample.get("id"))
        ids.append(sample_id)
        ground_truths.append(
            {
                "id": sample_id,
                "has_hidden_text": not bool(sample.get("is_white_sample", False)),
                "hidden_content": sample.get("hidden_content", ""),
                "category": meta_category(sample),
            }
        )
    return ids, ground_truths


def prediction_text(record: dict) -> str:
    if record.get("prediction_raw"):
        return record["prediction_raw"]
    if record.get("prediction"):
        return record["prediction"]
    if record.get("ocr_text") is not None:
        text = str(record.get("ocr_text") or "").strip()
        if text:
            return f"<has_hidden_text>Yes</has_hidden_text><hidden_content>{text}</hidden_content>"
        return "<has_hidden_text>No</has_hidden_text><hidden_content>None</hidden_content>"
    parsed = record.get("prediction_parsed")
    if isinstance(parsed, dict):
        flag = "Yes" if parsed.get("has_hidden_text") else "No"
        content = parsed.get("hidden_content") or "None"
        return f"<has_hidden_text>{flag}</has_hidden_text><hidden_content>{content}</hidden_content>"
    return ""


def load_predictions(path: Path) -> dict[str, str]:
    predictions = {}
    for record in load_jsonl(path):
        if "id" in record:
            predictions[str(record["id"])] = prediction_text(record)
    return predictions


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path)
    path = Path(spec)
    return path.parent.name or path.stem, path


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "threshold",
        "model",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
        "text_match_score",
        "num_samples",
        "num_predictions",
        "missing_predictions",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_markdown(rows: list[dict], thresholds: list[float], model_names: list[str], path: Path) -> None:
    by_model = {(row["model"], row["threshold"]): row for row in rows}
    lines = []
    lines.append("| Model | " + " | ".join(f"tau={tau:g}" for tau in thresholds) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(thresholds)) + " |")
    for model in model_names:
        cells = []
        for tau in thresholds:
            row = by_model[(model, tau)]
            cells.append(f"{100 * row['text_match_score']:.2f}")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep NED tolerance thresholds tau for one or more ImpText result files.")
    parser.add_argument("--dataset", default="imptext_bench/dataset.jsonl")
    parser.add_argument(
        "--results",
        action="append",
        required=True,
        help="Result file as name=path or just path. Can be repeated.",
    )
    parser.add_argument("--thresholds", default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated NED tolerance thresholds tau.")
    parser.add_argument("--output-dir", default="outputs/tau_sweep")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    ids, ground_truths = build_ground_truths(Path(args.dataset))
    specs = [parse_model_spec(spec) for spec in args.results]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, result_path in specs:
        predictions_by_id = load_predictions(result_path)
        ordered_predictions = [predictions_by_id.get(sample_id, "") for sample_id in ids]
        for threshold in thresholds:
            evaluator = Evaluator(match_threshold=threshold)
            metrics = evaluator.evaluate(ordered_predictions, ground_truths)
            rows.append(
                {
                    "threshold": threshold,
                    "model": model_name,
                    "f1_score": metrics.get("f1_score", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "text_match_score": metrics.get("text_match_score", 0.0),
                    "num_samples": metrics.get("num_samples", len(ground_truths)),
                    "num_predictions": len(predictions_by_id),
                    "missing_predictions": sum(1 for sample_id in ids if sample_id not in predictions_by_id),
                    "categories": metrics.get("categories", {}),
                }
            )

    model_names = [name for name, _ in specs]
    (output_dir / "tau_sweep.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows, output_dir / "tau_sweep.csv")
    write_markdown(rows, thresholds, model_names, output_dir / "tau_sweep.md")
    print(f"Wrote {output_dir / 'tau_sweep.json'}")
    print(f"Wrote {output_dir / 'tau_sweep.csv'}")
    print(f"Wrote {output_dir / 'tau_sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
