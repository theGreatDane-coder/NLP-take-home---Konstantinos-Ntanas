from typing import List, Dict
from data import RagRow
from scorers import SCORERS

class Judge:
    
    """
    Orchestrates evaluation of RAG rows using provided scorer definitions.
    """
    def __init__(self, scorers: List[Dict], mode: str = "heuristic"):
        """
        :param scorers: List of scorer descriptors (name, max_score, weight, heuristic_fn, llm_fn)
        :param mode: "heuristic" to use heuristic_fn, "llm" to use llm_fn
        """
        self.scorers = scorers
        if mode not in {"heuristic", "llm"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def compute_composite(self, scores: Dict[str, int]) -> float:
        """
        Compute weighted composite score normalized by max_scores.
        :param scores: dict mapping dimension name to raw integer score
        :return: composite score (0.0-1.0)
        """
        total = 0.0
        for sc in self.scorers:
            name = sc["name"]
            weight = sc["weight"]
            max_score = sc["max_score"]
            raw = scores.get(name, 0)
            total += weight * (raw / max_score)
        return total

    def evaluate_row(self, row: RagRow) -> Dict[str, float]:
        """
        Evaluate a single RAG row across all dimensions.
        :param row: RagRow instance
        :return: dict of raw scores and composite under key "Composite"
        """
        scores: Dict[str, int] = {}
        for sc in self.scorers:
            fn = sc["heuristic_fn"] if self.mode == "heuristic" else sc["llm_fn"]
            raw_score = fn(row)
            scores[sc["name"]] = raw_score

        composite = self.compute_composite(scores)
        scores["Composite"] = composite
        return scores

    def evaluate_all(self, rows: List[RagRow]) -> List[Dict]:
        """
        Evaluate a list of RAG rows, returning a list of result dicts.
        Each result dict includes original fields plus per-dimension scores and composite.
        """
        results: List[Dict] = []
        for row in rows:
            eval_scores = self.evaluate_row(row)
            entry: Dict = {
                "Current User Question": row.question,
                "Conversation History": row.history,
                "Fragment Texts": "\n".join(row.fragments),
                "Assistant Answer": row.answer,
            }

            # Merge scores
            entry.update(eval_scores)
            results.append(entry)
        return results
