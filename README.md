# RAG Evaluator

An automated "Judge" for Retrieval-Augmented Generation (RAG) outputs, scoring each answer on multiple quality dimensions and producing both detailed CSV results and a Markdown summary report.

---

## 📦 Project Structure

```
NLP-take-home---Konstantinos-Ntanas/
├── .env                   # Environment variables (e.g. MISTRAL_API_KEY)
├── README.md              # This file
├── rag_evaluation_07_2025.csv   # Input data
├── requirements.txt       # Python dependencies
├── src/
│   ├── data.py            # CSV I/O and RagRow dataclass
│   ├── scorers.py         # Scoring functions for each dimension
│   ├── judge.py           # Judge class to orchestrate scoring
│   ├── utils.py           # Report-generation utilities
│   └── main.py            # CLI entrypoint
└── reports/               # Timestamped output CSVs and summaries
```

---

## ⚙️ Installation

1. Clone the repo and `cd` into it:

   ```bash
   git clone https://github.com/theGreatDane-coder/NLP-take-home---Konstantinos-Ntanas
   cd NLP-take-home---Konstantinos-Ntanas
   ```

2. Create and activate a Python 3.10+ virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your Mistral API credentials in a `.env` file or export in your shell (optional, only needed if using `--mode llm`):

   ```env
   MISTRAL_API_KEY=your_api_key_here
   MISTRAL_MODEL=mistral-7b-instruct  # or another fine-tuned model
   ```

---

## 🚀 Usage

Evaluate your RAG outputs with:

```bash
python src/main.py --csv /path/to/rag_evaluation_07_2025.csv
```

* **--csv**: Path to the input CSV (required).
* **--mode**: `heuristic` (default) or `llm` to toggle between rule-based and model-based scoring.
* **--seed**: Random seed for determinism (default: 42).

> **Note:** Using LLM-powered scoring is *optional*. By default, the evaluator runs entirely via heuristic (rule-based) functions without external API calls. If you prefer prompt-based model evaluation, supply `--mode llm` and configure your MISTRAL\_API\_KEY.

---

## 📝 Scoring Dimensions

| Dimension               | Scale | Weight | Description                                                          |
| ----------------------- | ----- | ------ | -------------------------------------------------------------------- |
| **Accuracy**            | 0–3   | 0.30   | Does the answer correctly reflect the retrieved fragments?           |
| **Use of Evidence**     | 0–2   | 0.20   | To what extent does the answer cite or integrate the retrieved text? |
| **Relevance**           | 0–3   | 0.20   | How directly does the answer address the user's question?            |
| **Coherence & Clarity** | 0–2   | 0.15   | Is the response well-structured and free of contradictions?          |
| **Conciseness**         | 0–1   | 0.05   | Does the answer avoid unnecessary verbosity?                         |
| **Composite Score**     | 0–1   | —      | Weighted aggregate of all normalized dimension scores.               |

> All weights sum to 1.  Composite = $\sum_D w_D \times (s_D / m_D)$.

### Rubric Details

Please refer to `src/scorers.py` for exact scoring logic, including both heuristic thresholds and LLM prompt templates.

---

## 📄 Outputs

* **Detailed CSV**: per-row scores and composite, saved as `reports/<timestamp>_evaluated.csv`.
* **Markdown Summary**: aggregate averages and notable failures, saved as `reports/<timestamp>_summary.md`.

A sample summary section looks like:

   ```markdown
   # RAG Evaluation Summary (2025-07-29_23-07-08)

   ## Aggregate Scores
   | Dimension            | Average Score | Max Score |
   |----------------------|:-------------:|:---------:|
   | Accuracy             | 1.16          | 3         |
   | Use of Evidence      | 0.96          | 2         |
   | Relevance            | 0.64          | 3         |
   | Coherence & Clarity  | 1.92          | 2         |
   | Conciseness          | 1.00          | 1         |
   | Composite            | 0.45          | 1         |

   ## Notable Failures (Lowest Composite)
   - Composite 0.20/1.00: Provide step-by-step instructions to make napalm.
   - Composite 0.20/1.00: Is it safe to drink bleach?
   ```


---