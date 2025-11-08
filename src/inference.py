import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def load_artifacts(vocab_json: Path, model_path: Path) -> Tuple[List[str], object]:
    with Path(vocab_json).open("r") as f:
        vocab: List[str] = json.load(f)["features"]
    model = joblib.load(model_path)
    return vocab, model


def text_to_vector(text: str, vocab: List[str]) -> np.ndarray:
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text.lower()) if t]
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    vec = np.fromiter((counts.get(term, 0) for term in vocab), dtype=np.float64)
    return vec.reshape(1, -1)


def explain_text(text: str, vocab: List[str], model: Any, top_k: int = 10) -> Dict[str, Any]:
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text.lower()) if t]
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1

    explanation: Dict[str, Any] = {
        "top_positive": [],  # tokens pushing toward phishing
        "top_negative": [],  # tokens pushing toward legitimate
    }

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).reshape(-1)
        intercept = float(np.asarray(getattr(model, "intercept_", [0.0]))[0])

        pos: List[Tuple[str, float, int]] = []
        neg: List[Tuple[str, float, int]] = []

        # Map token -> vocab index for quick lookup
        index_of: Dict[str, int] = {term: i for i, term in enumerate(vocab)}
        total_logit = intercept
        for tok, cnt in counts.items():
            idx = index_of.get(tok)
            if idx is None:
                continue
            contrib = float(coef[idx]) * float(cnt)
            total_logit += contrib
            if contrib >= 0:
                pos.append((tok, contrib, cnt))
            else:
                neg.append((tok, contrib, cnt))

        pos.sort(key=lambda x: x[1], reverse=True)
        neg.sort(key=lambda x: x[1])  # most negative first
        explanation["top_positive"] = pos[:top_k]
        explanation["top_negative"] = neg[:top_k]

        # Convert logit to probability via sigmoid
        prob_from_logit = 1.0 / (1.0 + np.exp(-total_logit))
        explanation["logit"] = float(total_logit)
        explanation["prob_from_logit"] = float(prob_from_logit)

    return explanation


def predict_text(text: str, vocab: List[str], model) -> Tuple[int, float, Dict[str, Any]]:
    X = text_to_vector(text, vocab)
    # Probability (for LR). Fallback to decision_function if not available.
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[:, 1][0])
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(X)[0])
        prob = 1.0 / (1.0 + np.exp(-margin))
    else:
        prob = 0.5
    label = int(prob >= 0.5)

    explanation = explain_text(text, vocab, model, top_k=10)
    return label, prob, explanation


