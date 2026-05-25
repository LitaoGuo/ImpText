#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an ImpText-Bench manifest against local image files.")
    parser.add_argument("--dataset", default="imptext_bench/dataset.jsonl")
    parser.add_argument("--image-root", default="imptext_bench/images")
    parser.add_argument("--write-available", default=None, help="Optional path for records whose images exist.")
    parser.add_argument("--write-missing", default=None, help="Optional path for missing image paths.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    image_root = Path(args.image_root)
    records = load_jsonl(dataset_path)

    missing = []
    available = []
    categories = Counter()
    splits = Counter()
    for record in records:
        image_path = record["image_path"]
        categories[record.get("category", "Unknown")] += 1
        splits[image_path.split("/", 1)[0]] += 1
        if (image_root / image_path).exists():
            available.append(record)
        else:
            missing.append(image_path)

    print(f"dataset: {dataset_path}")
    print(f"image_root: {image_root}")
    print(f"records: {len(records)}")
    print(f"available_images: {len(available)}")
    print(f"missing_images: {len(missing)}")
    print("splits:")
    for key, value in sorted(splits.items()):
        print(f"  {key}: {value}")
    print("categories:")
    for key, value in sorted(categories.items()):
        print(f"  {key}: {value}")
    if missing:
        print("first_missing:")
        for path in missing[:10]:
            print(f"  {path}")

    if args.write_available:
        Path(args.write_available).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in available),
            encoding="utf-8",
        )
    if args.write_missing:
        Path(args.write_missing).write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

