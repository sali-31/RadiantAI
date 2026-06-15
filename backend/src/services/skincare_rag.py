import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "skincare_knowledge.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "should",
    "skin",
    "skincare",
    "the",
    "to",
    "use",
    "what",
    "with",
}


def tokenize(text: str) -> List[str]:
    tokens = []
    for token in re.split(r"[^a-z0-9]+", str(text or "").lower()):
        if len(token) <= 2 or token in STOPWORDS:
            continue
        tokens.append(token)
        if token.endswith("s") and len(token) > 4:
            tokens.append(token[:-1])
    return tokens


def vectorize(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_similarity(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class SkincareRAG:
    """Retrieve skincare knowledge chunks with local cosine similarity."""

    def __init__(self, data_path: Path = DATA_PATH) -> None:
        self.data_path = data_path
        self.documents = self._load_documents()
        self.document_vectors = [
            vectorize(
                " ".join(
                    [
                        document.get("title", ""),
                        " ".join(document.get("intents", [])),
                        " ".join(document.get("concerns", [])),
                        document.get("text", ""),
                    ]
                )
            )
            for document in self.documents
        ]

    def retrieve(self, message: str, nlu: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        query_text = " ".join(
            [
                message,
                str(nlu.get("intent", "")),
                " ".join(nlu.get("concerns") or []),
                str(nlu.get("answer_type", "")),
                str(nlu.get("product_form") or ""),
            ]
        )
        query_vector = vectorize(query_text)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        nlu_concerns = set(nlu.get("concerns") or [])
        nlu_intent = nlu.get("intent")

        for document, document_vector in zip(self.documents, self.document_vectors):
            score = cosine_similarity(query_vector, document_vector)
            doc_concerns = set(document.get("concerns") or [])
            doc_intents = set(document.get("intents") or [])
            if nlu_concerns & doc_concerns:
                score += 0.35
            if nlu_intent in doc_intents:
                score += 0.2
            if "general_skincare" in nlu_concerns and "general_skincare" in doc_concerns:
                score += 0.15
            scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, document in scored[:top_k]:
            result = dict(document)
            result["score"] = round(score, 4)
            results.append(result)
        return results

    def context_text(self, message: str, nlu: Dict[str, Any], top_k: int = 5) -> str:
        chunks = self.retrieve(message, nlu, top_k=top_k)
        if not chunks:
            return ""
        lines = []
        for index, chunk in enumerate(chunks, start=1):
            lines.append(f"{index}. {chunk['title']} [{chunk['id']}]: {chunk['text']}")
        return "\n".join(lines)

    def _load_documents(self) -> List[Dict[str, Any]]:
        raw = json.loads(self.data_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [document for document in raw if isinstance(document, dict)]
