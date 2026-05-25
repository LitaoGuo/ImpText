#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imptext.engines import APIEngine
from imptext.evaluation import Evaluator


OCR_PROMPT = "Extract any hidden or visually embedded text from the image. Output only the recovered text. If no such text is present, output nothing."


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


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


def resolve_image_path(image_root: Path, image_path: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str(image_root / p)


def wrap_ocr_text(text: str) -> str:
    content = (text or "").strip()
    if content:
        return f"<has_hidden_text>Yes</has_hidden_text><hidden_content>{content}</hidden_content>"
    return "<has_hidden_text>No</has_hidden_text><hidden_content>None</hidden_content>"


def parse_headers(items: list[str] | None) -> dict[str, str]:
    headers = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Header must use KEY=VALUE format: {item}")
        key, value = item.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def extract_path_values(obj: Any, path: str) -> list[Any]:
    parts = [part for part in path.split(".") if part]

    def walk(value: Any, index: int) -> list[Any]:
        if index == len(parts):
            return [value]
        part = parts[index]
        values = []
        if part == "*":
            if isinstance(value, list):
                for item in value:
                    values.extend(walk(item, index + 1))
            return values
        if isinstance(value, list):
            try:
                return walk(value[int(part)], index + 1)
            except (ValueError, IndexError):
                return []
        if isinstance(value, dict) and part in value:
            return walk(value[part], index + 1)
        return []

    return walk(obj, 0)


def extract_response_text(obj: Any, response_fields: list[str]) -> str:
    if isinstance(obj, str):
        return obj
    fields = response_fields or [
        "text",
        "result.text",
        "data.text",
        "result.markdown.text",
        "result.ocrResults.*.prunedResult",
        "result.layoutParsingResults.*.markdown.text",
    ]
    chunks = []
    for field in fields:
        for value in extract_path_values(obj, field):
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                chunks.append(json.dumps(value, ensure_ascii=False))
            else:
                chunks.append(str(value))
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def call_http_json(item: dict, args: argparse.Namespace, headers: dict[str, str]) -> dict:
    import requests

    with open(item["image_path"], "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    payload = {
        args.image_field: encoded,
        "file": encoded,
        "fileType": 1,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }
    response = requests.post(args.http_url, json=payload, headers=headers, timeout=args.timeout)
    response.raise_for_status()
    try:
        body = response.json()
    except ValueError:
        body = response.text
    text = extract_response_text(body, args.response_field)
    return {
        "id": item["id"],
        "text": text,
        "success": True,
        "last_error": None,
        "status_code": response.status_code,
    }


def build_metrics(records: list[dict], threshold: float) -> dict:
    evaluator = Evaluator(match_threshold=threshold)
    implicit_records = [record for record in records if record["ground_truth"]["has_hidden_text"]]
    category_scores: dict[str, list[int]] = {}
    category_sims: dict[str, list[float]] = {}
    scores = []
    similarities = []
    for record in implicit_records:
        gt = record["ground_truth"]
        sim = evaluator.compute_text_similarity(record.get("ocr_text", ""), gt["hidden_content"])
        score = evaluator.compute_text_match_score(record.get("ocr_text", ""), gt["hidden_content"])
        scores.append(score)
        similarities.append(sim)
        category_scores.setdefault(gt["category"], []).append(score)
        category_sims.setdefault(gt["category"], []).append(sim)

    wrapped_predictions = [wrap_ocr_text(record.get("ocr_text", "")) for record in records]
    ground_truths = [record["ground_truth"] for record in records]
    classification_metrics = evaluator.evaluate(wrapped_predictions, ground_truths)

    return {
        "threshold": threshold,
        "ocr_text_match_score": sum(scores) / len(scores) if scores else 0.0,
        "mean_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
        "num_samples": len(records),
        "num_implicit_samples": len(implicit_records),
        "classification_if_nonempty_is_hidden": classification_metrics,
        "categories": {
            category: {
                "ocr_text_match_score": sum(values) / len(values) if values else 0.0,
                "mean_similarity": sum(category_sims[category]) / len(category_sims[category]) if category_sims[category] else 0.0,
                "num_samples": len(values),
            }
            for category, values in sorted(category_scores.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or evaluate OCR-style baselines on ImpText-Bench.")
    parser.add_argument("--engine", choices=["openai", "http-json"], default="openai")
    parser.add_argument("--model", default="ocr-baseline", help="Model name for openai engine or output label.")
    parser.add_argument("--dataset", default="imptext_bench/dataset.jsonl")
    parser.add_argument("--image-root", default="imptext_bench/images")
    parser.add_argument("--output-dir", default="outputs/ocr_eval")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5, help="NED tolerance threshold tau for TMS. Default: 0.5.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without initializing clients or sending requests.")
    parser.add_argument("--prompt", default=OCR_PROMPT)
    parser.add_argument("--http-url", default=None)
    parser.add_argument("--header", action="append", help="Extra HTTP header in KEY=VALUE format. Can be repeated.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable containing an HTTP API key.")
    parser.add_argument("--api-key-header", default="Authorization")
    parser.add_argument("--api-key-prefix", default="Bearer")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--response-field", action="append", help="Dot path for OCR text in JSON response. Supports '*' for lists.")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir) / safe_name(args.model)
    results_path = output_dir / "results.jsonl"
    metrics_path = output_dir / "metrics.json"

    dataset = load_jsonl(dataset_path)
    if args.limit > 0:
        dataset = dataset[: args.limit]

    processed = set()
    if results_path.exists():
        for record in load_jsonl(results_path):
            processed.add(str(record.get("id")))

    items = []
    missing = []
    for sample in dataset:
        sample_id = str(sample.get("id"))
        image_path = resolve_image_path(image_root, sample["image_path"])
        if not os.path.exists(image_path):
            missing.append({"id": sample_id, "image_path": image_path})
            if args.skip_missing:
                continue
        if sample_id in processed:
            continue
        items.append({"id": sample_id, "image_path": image_path, "sample": sample, "prompt": args.prompt})

    if missing and not args.skip_missing:
        print(f"Missing {len(missing)} images. First missing: {missing[0]['image_path']}", file=sys.stderr)
        return 2
    if args.engine == "http-json" and not args.http_url and not args.dry_run:
        print("--http-url is required for --engine http-json", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Engine: {args.engine}")
        print(f"Dataset records after limit: {len(dataset)}")
        print(f"Images missing: {len(missing)}")
        print(f"Already processed: {len(processed)}")
        print(f"Samples that would be processed: {len(items)}")
        print("Dry run complete. No requests were sent.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.engine == "openai":
        engine = APIEngine(model=args.model, concurrency=args.concurrency)
        with results_path.open("a", encoding="utf-8") as f:
            for item, result in engine.generate(items):
                sample = item["sample"]
                gt = build_ground_truth(sample)
                ocr_text = result["text"].strip()
                record = {
                    "id": item["id"],
                    "category": gt["category"],
                    "image_path": sample["image_path"],
                    "ground_truth": gt,
                    "ocr_text": ocr_text,
                    "prediction_raw": wrap_ocr_text(ocr_text),
                    "request_meta": {
                        "engine": args.engine,
                        "success": result["success"],
                        "attempts": result["attempts"],
                        "last_error": result["last_error"],
                    },
                    "token_usage": result["usage"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    else:
        headers = parse_headers(args.header)
        api_key = os.getenv(args.api_key_env) if args.api_key_env else None
        if api_key:
            value = f"{args.api_key_prefix} {api_key}".strip() if args.api_key_prefix else api_key
            headers[args.api_key_header] = value
        with results_path.open("a", encoding="utf-8") as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {executor.submit(call_http_json, item, args, headers): item for item in items}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="OCR HTTP inference"):
                    item = futures[future]
                    sample = item["sample"]
                    gt = build_ground_truth(sample)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"id": item["id"], "text": "", "success": False, "last_error": str(exc), "status_code": None}
                    ocr_text = result["text"].strip()
                    record = {
                        "id": item["id"],
                        "category": gt["category"],
                        "image_path": sample["image_path"],
                        "ground_truth": gt,
                        "ocr_text": ocr_text,
                        "prediction_raw": wrap_ocr_text(ocr_text),
                        "request_meta": {
                            "engine": args.engine,
                            "success": result["success"],
                            "last_error": result["last_error"],
                            "status_code": result["status_code"],
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

    records = load_jsonl(results_path)
    metrics = build_metrics(records, args.threshold)
    metrics["model"] = args.model
    metrics["engine"] = args.engine
    metrics["timestamp"] = dt.datetime.now().isoformat()
    metrics["results_path"] = str(results_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Results: {results_path}")
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
