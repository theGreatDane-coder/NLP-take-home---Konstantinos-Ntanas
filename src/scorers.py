import os
import requests
from data import RagRow
from dotenv import load_dotenv

load_dotenv()

# Configuration for Mistral API
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-7b-instruct")
MISTRAL_ENDPOINT = f"https://api.mistral.ai/v1/models/{MISTRAL_MODEL}/completions"

HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

# Helper: call Mistral completion
def call_mistral(prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 16,
        "n": 1,
        "stop": ["\n"]
    }
    response = requests.post(MISTRAL_ENDPOINT, headers=HEADERS, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["text"].strip()


# Scoring Functions for Each Dimension

def score_accuracy_heuristic(row: RagRow) -> int:
    frag_tokens = set(" ".join(row.fragments).lower().split())
    ans_tokens = set(row.answer.lower().split())
    overlap = len(frag_tokens & ans_tokens)
    if overlap < 3:
        return 0
    if overlap < 6:
        return 1
    if overlap < 10:
        return 2
    return 3


def score_accuracy_llm(row: RagRow) -> int:
    prompt = f"""
    You are an expert evaluator. Here are source passages:
    {chr(10).join(row.fragments)}

    Answer:
    {row.answer}

    Using this rubric:
    0 = Contains factual errors.
    1 = Mostly correct but has small mistakes.
    2 = Correct on major points but misses minor detail.
    3 = Perfectly matches all information.

    Return only the integer in plain text.
    """
    result = call_mistral(prompt)
    return int(result)


def score_evidence_heuristic(row: RagRow) -> int:
    frag_tokens = set(" ".join(row.fragments).lower().split())
    ans_tokens = set(row.answer.lower().split())
    overlap = len(frag_tokens & ans_tokens)
    if overlap == 0:
        return 0
    if "[" in row.answer and "]" in row.answer:
        return 2
    return 1


def score_evidence_llm(row: RagRow) -> int:
    prompt = f"""
    You are an expert evaluator. Here are source passages:
    {chr(10).join(row.fragments)}

    Answer:
    {row.answer}

    Using this rubric:
    0 = No use of the retrieved passages.
    1 = Paraphrases or mentions but no formal citation.
    2 = Clearly cites or integrates passages.

    Return only the integer in plain text.
    """
    result = call_mistral(prompt)
    return int(result)


def score_relevance_heuristic(row: RagRow) -> int:
    qs_tokens = set(row.question.lower().split())
    ans_tokens = set(row.answer.lower().split())
    overlap = len(qs_tokens & ans_tokens)
    if overlap < 2:
        return 0
    if overlap < 4:
        return 1
    if overlap < 7:
        return 2
    return 3


def score_relevance_llm(row: RagRow) -> int:
    prompt = f"""
    You are an expert evaluator. User question:
    {row.question}

    Answer:
    {row.answer}

    Using this rubric:
    0 = Off-topic.
    1 = Partially addresses.
    2 = Addresses but misses nuance.
    3 = Fully and directly addresses the user's question.

    Return only the integer in plain text.
    """
    result = call_mistral(prompt)
    return int(result)


def score_coherence_heuristic(row: RagRow) -> int:
    sentences = [s for s in row.answer.split('.') if s.strip()]
    if not sentences:
        return 0
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_len < 5 or avg_len > 30:
        return 1
    return 2


def score_coherence_llm(row: RagRow) -> int:
    prompt = f"""
    You are an expert evaluator. Evaluate the coherence and clarity of this answer:
    {row.answer}

    Using this rubric:
    0 = Disjointed or contradictory.
    1 = Understandable but awkward.
    2 = Very clear and well structured.

    Return only the integer in plain text.
    """
    result = call_mistral(prompt)
    return int(result)


def score_conciseness_heuristic(row: RagRow) -> int:
    word_count = len(row.answer.split())
    return 1 if word_count <= 50 else 0


def score_conciseness_llm(row: RagRow) -> int:
    prompt = f"""
    You are an expert evaluator. Evaluate the conciseness of this answer:
    {row.answer}

    Using this rubric:
    0 = Verbose or repetitive.
    1 = Succinct, every sentence adds value.

    Return only the integer in plain text.
    """
    result = call_mistral(prompt)
    return int(result)


# Registry of All Scorers
SCORERS = [
    {"name": "Accuracy",            "max_score": 3, "weight": 0.30, "heuristic_fn": score_accuracy_heuristic,    "llm_fn": score_accuracy_llm},
    {"name": "Use of Evidence",     "max_score": 2, "weight": 0.20, "heuristic_fn": score_evidence_heuristic,    "llm_fn": score_evidence_llm},
    {"name": "Relevance",           "max_score": 3, "weight": 0.20, "heuristic_fn": score_relevance_heuristic,   "llm_fn": score_relevance_llm},
    {"name": "Coherence & Clarity", "max_score": 2, "weight": 0.15, "heuristic_fn": score_coherence_heuristic,   "llm_fn": score_coherence_llm},
    {"name": "Conciseness",         "max_score": 1, "weight": 0.05, "heuristic_fn": score_conciseness_heuristic, "llm_fn": score_conciseness_llm},
]
