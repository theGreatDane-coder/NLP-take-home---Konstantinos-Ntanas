from dataclasses import dataclass
from typing import List, Dict
import csv

@dataclass
class RagRow:
    """
    Represents a single RAG evaluation example:
      - question: the user's current question
      - history: conversation history as a single string (newline-delimited)
      - fragments: list of retrieved text passages
      - answer: the assistant's generated answer
    """
    question: str
    history: str
    fragments: List[str]
    answer: str


def load_csv(path: str) -> List[RagRow]:
    """
    Load the RAG evaluation CSV and return a list of RagRow objects.
    Expects columns:
      - Current User Question
      - Conversation History
      - Fragment Texts (newline-separated)
      - Assistant Answer
    """
    rows: List[RagRow] = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for record in reader:
            raw_frags = record.get('Fragment Texts', '').split('\n')
            frags = [f.strip() for f in raw_frags if f.strip()]
            rows.append(
                RagRow(
                    question=record.get('Current User Question', '').strip(),
                    history=record.get('Conversation History', '').strip(),
                    fragments=frags,
                    answer=record.get('Assistant Answer', '').strip()
                )
            )
    return rows


def save_results(results: List[Dict], path: str = "evaluated.csv") -> None:
    """
    Save evaluation results to a CSV file. `results` should be a list of dicts,
    where each dict contains original RagRow fields plus score columns and CompositeScore.
    If `path` is omitted, defaults to 'evaluated.csv'.
    """
    if not results:
        print(f"No results to save to {path}.")
        return
    fieldnames = list(results[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
