import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "imptext_bench" / "dataset.jsonl"


class ManifestTest(unittest.TestCase):
    def test_manifest_schema_and_counts(self):
        records = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(records), 1630)
        required = {"id", "category", "image_path", "is_white_sample", "hidden_content"}
        self.assertTrue(all(required <= set(record) for record in records))
        self.assertEqual(sum(record["is_white_sample"] for record in records), 1141)
        self.assertEqual(sum(not record["is_white_sample"] for record in records), 489)


if __name__ == "__main__":
    unittest.main()

