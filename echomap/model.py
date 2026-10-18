from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_SCHEMA


@dataclass
class TrainingExample:
    features: np.ndarray
    label: str
    session: str


class PresenceModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = HistGradientBoostingClassifier(max_iter=100, random_state=7)
        self.labels: list[str] = []
        self.settings: dict = {}

    def fit(self, examples: list[TrainingExample], settings: dict | None = None) -> dict:
        if len({item.label for item in examples}) < 2:
            raise ValueError("Training requires at least two labels")
        sessions = list(dict.fromkeys(item.session for item in examples))
        if len(sessions) < 2:
            raise ValueError("Training requires at least two recording sessions")
        holdout_session = sessions[-1]
        train = [item for item in examples if item.session != holdout_session]
        test = [item for item in examples if item.session == holdout_session]
        if len({item.label for item in train}) < 2:
            raise ValueError("The training-session split must contain at least two labels")
        x_train = self.scaler.fit_transform([item.features for item in train])
        self.classifier.fit(x_train, [item.label for item in train])
        self.labels = list(self.classifier.classes_)
        self.settings = {"feature_schema": FEATURE_SCHEMA, **(settings or {})}
        if test:
            predicted = self.classifier.predict(self.scaler.transform([item.features for item in test]))
            truth = [item.label for item in test]
            matrix = confusion_matrix(truth, predicted, labels=self.labels)
            precision, recall, _, _ = precision_recall_fscore_support(truth, predicted, labels=self.labels, zero_division=0)
            return {"holdout_session": holdout_session, "labels": self.labels, "confusion_matrix": matrix.tolist(), "precision": precision.tolist(), "recall": recall.tolist(), "test_count": len(test)}
        return {"test_count": 0}

    def predict(self, features: np.ndarray, minimum_margin: float = 0.15) -> tuple[str, float, bool]:
        if not self.labels:
            return "uncertain", 0.0, False
        transformed = self.scaler.transform([features])
        probabilities = self.classifier.predict_proba(transformed)[0]
        best = int(np.argmax(probabilities))
        ordered = np.sort(probabilities)
        margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0
        confidence = float(probabilities[best])
        return (self.labels[best], confidence, confidence >= 0.55 and margin >= minimum_margin)

    def save(self, path: str | Path) -> None:
        payload = {"model": self.classifier, "scaler": self.scaler, "labels": self.labels, "settings": self.settings, "feature_schema": FEATURE_SCHEMA}
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "PresenceModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        result = cls()
        result.classifier = payload["model"]
        result.scaler = payload["scaler"]
        result.labels = payload["labels"]
        result.settings = payload["settings"]
        return result


def save_examples(path: str | Path, examples: list[TrainingExample]) -> None:
    rows = [{"features": item.features.tolist(), "label": item.label, "session": item.session} for item in examples]
    Path(path).write_text(json.dumps(rows, indent=2))


def load_examples(path: str | Path) -> list[TrainingExample]:
    return [TrainingExample(np.asarray(row["features"], dtype=float), row["label"], row["session"]) for row in json.loads(Path(path).read_text())]
