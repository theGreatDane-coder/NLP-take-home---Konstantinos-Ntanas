import os
import datetime
from typing import List, Dict
from scorers import SCORERS


def generate_markdown_summary(results: List[Dict], summary_path: str) -> None:
    """
    Generate a Markdown summary report including aggregate scores and notable failures.

    :param results: List of dictionaries containing evaluation results per row.
    :param summary_path: Path to write the Markdown file.
    """
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(summary_path, "w", encoding="utf-8") as md:
        md.write(f"# RAG Evaluation Summary ({timestamp})\n\n")
        # Aggregate statistics
        md.write("## Aggregate Scores\n")
        md.write("| Dimension | Average Score | Max Score |\n")
        md.write("|-----------|:-------------:|:---------:|\n")
        # Calculate means per dimension
        for sc in SCORERS:
            name = sc['name']
            max_score = sc['max_score']
            values = [r.get(name, 0) for r in results]
            avg = sum(values) / len(values) if values else 0
            md.write(f"| {name} | {avg:.2f} | {max_score} |\n")
        # Composite
        comp_values = [r.get("Composite", 0) for r in results]
        comp_avg = sum(comp_values) / len(comp_values) if comp_values else 0
        md.write(f"| Composite | {comp_avg:.2f} | 1 |\n")
        md.write("\n")
        # Notable failures
        md.write("## Notable Failures (Lowest Composite)\n")
        sorted_results = sorted(results, key=lambda r: r.get("Composite", 0))
        for r in sorted_results[:5]:
            q = r.get("Current User Question", "")
            comp = r.get("Composite", 0)
            md.write(f"- Composite {comp:.2f}/1.00: {q}\n")
