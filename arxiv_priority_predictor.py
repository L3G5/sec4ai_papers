from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def normalize_text(text: str | None) -> str:
    text = "" if text is None else text
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_categories(text: str | None) -> str:
    text = "" if text is None else text
    tokens = [token.replace(".", "_").lower() for token in text.split()]
    return " ".join(tokens)


def extract_author_surnames(authors_parsed: bytes | str | None, authors: str | None = None) -> str:
    if authors_parsed is not None and authors_parsed != "":
        try:
            parsed = json.loads(authors_parsed)
        except (TypeError, json.JSONDecodeError):
            try:
                raw = authors_parsed.decode() if isinstance(authors_parsed, bytes) else authors_parsed
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError, AttributeError):
                parsed = []
        surnames = [normalize_text(item[0]) for item in parsed if item and item[0]]
        return " ".join(part for part in surnames if part)

    authors = "" if authors is None else authors
    rough_parts = re.split(r",| and ", authors)
    surnames = []
    for part in rough_parts:
        tokens = normalize_text(part).split()
        if tokens:
            surnames.append(tokens[-1])
    return " ".join(surnames)


@dataclass
class ArxivPriorityPredictor:
    model: Any
    metadata: dict[str, Any]

    def prepare_frame(self, records: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        for column in ["title", "abstract", "authors", "categories"]:
            if column not in frame.columns:
                frame[column] = ""
        if "authors_parsed" not in frame.columns:
            frame["authors_parsed"] = None

        prepared = frame.copy()
        prepared["title_text"] = prepared["title"].map(normalize_text)
        prepared["abstract_text"] = prepared["abstract"].map(normalize_text)
        prepared["author_text"] = [
            extract_author_surnames(authors_parsed, authors)
            for authors_parsed, authors in zip(prepared["authors_parsed"], prepared["authors"], strict=False)
        ]
        prepared["category_text"] = prepared["categories"].map(normalize_categories)
        return prepared

    def score(self, records: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        prepared = self.prepare_frame(records)
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(prepared)[:, 1]
        else:
            scores = self.model.decision_function(prepared)

        ranked = prepared.copy()
        ranked["score"] = np.asarray(scores, dtype=float)
        ranked["rank"] = ranked["score"].rank(method="first", ascending=False).astype(int)
        return ranked.sort_values("score", ascending=False).reset_index(drop=True)
