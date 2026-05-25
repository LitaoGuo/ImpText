import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from imptext.evaluation import Evaluator


class EvaluatorTest(unittest.TestCase):
    def test_parse_tagged_output(self):
        parsed = Evaluator().parse_output(
            "<has_hidden_text> Yes </has_hidden_text><hidden_content>abc123</hidden_content>"
        )
        self.assertTrue(parsed["has_hidden_text"])
        self.assertEqual(parsed["hidden_content"], "abc123")

    def test_text_match_threshold(self):
        evaluator = Evaluator(match_threshold=0.5)
        self.assertEqual(evaluator.compute_text_match_score("abc123", "abc123"), 1)
        self.assertEqual(evaluator.compute_text_match_score("", "abc123"), 0)

    def test_evaluate_minimal_batch(self):
        evaluator = Evaluator(match_threshold=0.5)
        metrics = evaluator.evaluate(
            [
                "<has_hidden_text>No</has_hidden_text><hidden_content>None</hidden_content>",
                "<has_hidden_text>Yes</has_hidden_text><hidden_content>abc123</hidden_content>",
            ],
            [
                {"has_hidden_text": False, "hidden_content": "", "category": "BENIGN SAMPLES"},
                {"has_hidden_text": True, "hidden_content": "abc123", "category": "VISUAL CAMOUFLAGE"},
            ],
        )
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["text_match_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
