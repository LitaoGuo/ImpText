import re
from typing import List, Dict, Any


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
    }

class Evaluator:
    def __init__(self, match_threshold: float = 0.5):
        self.match_threshold = match_threshold
        self.ned_threshold = match_threshold

    def parse_output(self, text: str) -> Dict[str, Any]:
        """
        Parse model output in XML-like format.
        Format: <think>...</think> <has_hidden_text>Yes/No</has_hidden_text> <hidden_content>...</hidden_content>
        """
        try:
            # Extract has_hidden_text
            has_hidden_match = re.search(r"<has_hidden_text>\s*(.*?)\s*</has_hidden_text>", text, re.IGNORECASE | re.DOTALL)
            has_hidden_text = False
            if has_hidden_match:
                content = has_hidden_match.group(1).strip().lower()
                if "yes" in content:
                    has_hidden_text = True
            
            # Extract hidden_content
            hidden_content_match = re.search(r"<hidden_content>\s*(.*?)\s*</hidden_content>", text, re.IGNORECASE | re.DOTALL)
            hidden_content = ""
            if hidden_content_match:
                hidden_content = hidden_content_match.group(1).strip()
                if hidden_content.lower() == "none":
                    hidden_content = ""
            
            return {
                "has_hidden_text": has_hidden_text,
                "hidden_content": hidden_content
            }
        except Exception:
            # Fallback for unexpected formats
            has_hidden = "yes" in text.lower() and "<has_hidden_text>" not in text # Prevent false positives if tag exists but failed regex
            # If we really failed to parse tags, maybe fallback to searching for "Yes" or "No"
            return {"has_hidden_text": False, "hidden_content": ""}

    def compute_normalized_edit_distance(self, pred: str, ref: str) -> float:
        """
        Compute normalized edit distance (NED).
        0.0 means exact match, 1.0 means maximally different.
        """
        if not pred and not ref:
            return 0.0
        if not pred or not ref:
            return 1.0
            
        # Normalize whitespace and case before computing edit distance.
        p = pred.lower().replace(" ", "")
        r = ref.lower().replace(" ", "")
        
        dist = _levenshtein_distance(p, r)
        max_len = max(len(p), len(r))
        if max_len == 0:
            return 0.0
        return dist / max_len

    def compute_text_similarity(self, pred: str, ref: str) -> float:
        """
        Compute normalized Levenshtein similarity.
        1.0 means exact match, 0.0 means maximally different.
        """
        return 1.0 - self.compute_normalized_edit_distance(pred, ref)
        
    def compute_text_match_score(self, pred: str, ref: str) -> int:
        """
        Binary Text Match Score. A prediction succeeds when NED <= tau.
        """
        ned = self.compute_normalized_edit_distance(pred, ref)
        return 1 if ned <= self.ned_threshold else 0

    def evaluate(self, predictions: List[str], ground_truths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute all metrics, including per-category metrics.
        """
        from collections import defaultdict
        
        # Global metric containers.
        global_y_true = []
        global_y_pred = []
        global_match_scores = []
        
        # Per-category metric containers.
        cat_y_true = defaultdict(list)
        cat_y_pred = defaultdict(list)
        cat_match_scores = defaultdict(list)
        
        for raw_pred, gt in zip(predictions, ground_truths):
            parsed = self.parse_output(raw_pred)
            category = gt.get("category", "Unknown")
            
            # 1. Existence classification.
            gt_cls = gt["has_hidden_text"]
            pred_cls = parsed["has_hidden_text"]
            
            global_y_true.append(gt_cls)
            global_y_pred.append(pred_cls)
            cat_y_true[category].append(gt_cls)
            cat_y_pred[category].append(pred_cls)
            
            # 2. Text match data for implicit samples.
            if gt_cls:
                score = self.compute_text_match_score(parsed["hidden_content"], gt["hidden_content"])
                
                global_match_scores.append(score)
                cat_match_scores[category].append(score)
        
        # Helper to calc metrics
        def _calc(y_t, y_p, matches):
            if not y_t: return {}
            summary = _binary_metrics(y_t, y_p)
            avg_match = sum(matches) / len(matches) if matches else 0.0
            
            result = {
                "f1_score": summary["f1_score"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "accuracy": summary["accuracy"],
                "text_match_score": float(avg_match),
                "num_samples": len(y_t)
            }
            return result

        # Global metrics.
        metrics = _calc(
            global_y_true, 
            global_y_pred, 
            global_match_scores
        )
        
        # Per-category metrics.
        metrics["categories"] = {}
        for cat in cat_y_true.keys():
            metrics["categories"][cat] = _calc(
                cat_y_true[cat], 
                cat_y_pred[cat], 
                cat_match_scores[cat]
            )

        # Paper-facing summary: benign samples are reported with accuracy,
        # while implicit categories are reported with Recall and TMS.
        paper_metrics = {
            "overall_f1": metrics.get("f1_score", 0.0),
            "overall_tms": metrics.get("text_match_score", 0.0),
            "benign_accuracy": None,
            "benign_num_samples": 0,
            "implicit_categories": {},
        }
        for cat, cat_metrics in metrics["categories"].items():
            is_benign = cat.upper() == "BENIGN SAMPLES" or not any(cat_y_true[cat])
            if is_benign:
                paper_metrics["benign_accuracy"] = cat_metrics.get("accuracy", 0.0)
                paper_metrics["benign_num_samples"] += cat_metrics.get("num_samples", 0)
            else:
                paper_metrics["implicit_categories"][cat] = {
                    "recall": cat_metrics.get("recall", 0.0),
                    "text_match_score": cat_metrics.get("text_match_score", 0.0),
                    "num_samples": cat_metrics.get("num_samples", 0),
                }
        metrics["paper_metrics"] = paper_metrics
            
        return metrics
