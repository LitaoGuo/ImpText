import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_benchmark.py"


class CheckBenchmarkTest(unittest.TestCase):
    def test_diagnostic_outputs_create_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            image_root = root / "images"
            image_path = image_root / "white" / "0001.png"
            output_dir = root / "nested" / "diagnostics"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"not-a-real-png")
            dataset.write_text(
                json.dumps(
                    {
                        "id": "0001",
                        "category": "Benign Samples",
                        "image_path": "white/0001.png",
                        "is_white_sample": True,
                        "hidden_content": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset",
                    str(dataset),
                    "--image-root",
                    str(image_root),
                    "--write-available",
                    str(output_dir / "available_records.jsonl"),
                    "--write-missing",
                    str(output_dir / "missing_image_paths.txt"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output_dir / "available_records.jsonl").exists())
            self.assertTrue((output_dir / "missing_image_paths.txt").exists())


if __name__ == "__main__":
    unittest.main()
