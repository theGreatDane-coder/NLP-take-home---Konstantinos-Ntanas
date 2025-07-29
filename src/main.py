import os
import argparse
import datetime
import random
from data import load_csv, save_results
from judge import Judge
from scorers import SCORERS
from utils import generate_markdown_summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG outputs with various scoring dimensions.")
    parser.add_argument(
        "--csv", dest="csv_path", required=True,
        help="Path to the input RAG evaluation CSV"
    )
    parser.add_argument(
        "--mode", dest="mode", choices=["heuristic", "llm"], default="heuristic",
        help="Scoring mode: 'heuristic' for rule-based, 'llm' for model-based"
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Seed RNGs for determinism
    random.seed(args.seed)

    # Load data
    rows = load_csv(args.csv_path)

    # Prepare reports directory and timestamp
    project_root = os.getcwd()
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Determine detailed CSV and summary paths in reports/
    csv_filename = f"{timestamp}_evaluated.csv"
    csv_path = os.path.join(reports_dir, csv_filename)

    summary_filename = f"{timestamp}_summary.md"
    summary_path = os.path.join(reports_dir, summary_filename)

    # Instantiate judge and evaluate
    judge = Judge(scorers=SCORERS, mode=args.mode)
    results = judge.evaluate_all(rows)

    # Save detailed results CSV
    save_results(results, csv_path)
    print(f"Detailed results written to {csv_path}")

    # Generate Markdown summary
    generate_markdown_summary(results, summary_path)
    print(f"Summary report written to {summary_path}")


if __name__ == "__main__":
    main()