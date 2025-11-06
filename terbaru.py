#!/usr/bin/env python3
"""
Hoax news detection pipeline.

This script loads all cleaned datasets from ``dataset/Cleaned``, performs
basic text normalisation, trains a lightweight classifier, evaluates the
model, and optionally runs an interactive prediction shell.  It is designed
to run locally without Google Colab specific features or ad-hoc dependency
installs.

Usage examples:
    python terbaru.py                # train model and save to models/hoax_detector.joblib
    python terbaru.py train --interactive
    python terbaru.py predict --text "Contoh berita yang ingin dicek."
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

try:
    from unidecode import unidecode  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    unidecode = None  # type: ignore

try:
    import emoji  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    emoji = None  # type: ignore

try:
    import joblib  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "Modul 'joblib' diperlukan. Instal terlebih dahulu, misal: pip install joblib"
    ) from exc


BASE_DIR = Path("dataset") / "Cleaned"
DEFAULT_MODEL_PATH = Path("models") / "hoax_detector.joblib"
HOAX_LABEL = 0
VALID_LABEL = 1
LABEL_NAMES = {HOAX_LABEL: "HOAKS", VALID_LABEL: "ASLI"}

SLANG_MAP = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "yg": "yang",
    "dr": "dari",
    "tp": "tapi",
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z0-9\s.,;:!?()'\-]")
LEAK_PATTERNS = [re.compile(p) for p in (r"\bhoaks?\b", r"\bhoax(es)?\b", r"\bturnbackhoax\b")]


@dataclass
class TrainingResult:
    model: Pipeline
    report: str
    confusion: np.ndarray
    accuracy: float
    test_samples: int


def normalise_text(text: str) -> str:
    """Basic normalisation: lowercase, remove URLs/mentions/emoji, map slang."""
    if not isinstance(text, str):
        return ""

    cleaned = text
    if unidecode:
        cleaned = unidecode(cleaned)
    cleaned = cleaned.lower()
    if emoji:
        cleaned = emoji.replace_emoji(cleaned, replace=" ")
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    for pattern in LEAK_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = NON_ALPHA_PATTERN.sub(" ", cleaned)
    cleaned = " ".join(SLANG_MAP.get(word, word) for word in cleaned.split())
    return re.sub(r"\s+", " ", cleaned).strip()


def _infer_text_column(df: pd.DataFrame) -> str:
    """Pick the most likely column containing textual content."""
    candidates = ["teks_bersih", "teks", "isi", "content", "text", "artikel", "isi_berita"]
    for name in candidates:
        if name in df.columns:
            return name
    return df.columns[0]


def _label_from_filename(stem: str) -> int:
    """Turnbackhoax dataset berisi contoh hoaks."""
    return HOAX_LABEL if "turnbackhoax" in stem.lower() else VALID_LABEL


def load_dataset(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """Load every Excel file in base_dir and return a cleaned DataFrame."""
    files = sorted(base_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"Tidak menemukan file .xlsx di {base_dir}. Pastikan dataset tersedia."
        )

    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_excel(path)
        text_column = _infer_text_column(df)
        cleaned = df[text_column].astype(str).map(normalise_text)
        stem = path.stem.lower()
        label = _label_from_filename(stem)
        frames.append(
            pd.DataFrame(
                {
                    "teks_asli": df[text_column].astype(str),
                    "teks_bersih": cleaned,
                    "sumber": stem,
                    "label": label,
                }
            )
        )

    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data.dropna(subset=["teks_bersih"])
    all_data = all_data[all_data["teks_bersih"].str.len() > 0]
    all_data = all_data.drop_duplicates(subset=["teks_bersih", "label"])
    return all_data.reset_index(drop=True)


def build_pipeline(class_weight: dict[int, float]) -> Pipeline:
    """Create a text classification pipeline using TF-IDF + Logistic Regression."""
    vectoriser = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        solver="lbfgs",
    )
    return Pipeline(
        [
            ("tfidf", vectoriser),
            ("clf", classifier),
        ]
    )


def train_model(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Split data, train the model, and compute evaluation metrics."""
    X = data["teks_bersih"].to_numpy()
    y = data["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

    pipeline = build_pipeline(weight_dict)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=[LABEL_NAMES[c] for c in classes], digits=4
    )
    confusion = confusion_matrix(y_test, y_pred, labels=[HOAX_LABEL, VALID_LABEL])
    accuracy = pipeline.score(X_test, y_test)

    return TrainingResult(
        model=pipeline,
        report=report,
        confusion=confusion,
        accuracy=accuracy,
        test_samples=len(y_test),
    )


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {path}. Jalankan 'python terbaru.py train' terlebih dahulu."
        )
    return joblib.load(path)


def predict_texts(model: Pipeline, texts: Sequence[str]) -> List[dict[str, object]]:
    """Return probability for each input text."""
    if not texts:
        return []
    cleaned_inputs = [normalise_text(text) for text in texts]
    probabilities = model.predict_proba(cleaned_inputs)[:, VALID_LABEL]
    predictions = model.predict(cleaned_inputs)
    results = []
    for original, cleaned, label, prob in zip(texts, cleaned_inputs, predictions, probabilities):
        results.append(
            {
                "original": original,
                "cleaned": cleaned,
                "label": LABEL_NAMES[int(label)],
                "probability_valid": float(prob),
            }
        )
    return results


def interactive_shell(model: Pipeline) -> None:
    print("\nMode interaktif deteksi hoaks.")
    print("Ketik teks berita, kosongkan baris untuk memproses, atau ketik 'exit' untuk keluar.\n")
    buffer: List[str] = []
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar dari mode interaktif.")
            return
        if line.strip().lower() == "exit":
            print("Keluar dari mode interaktif.")
            return
        if line.strip() == "":
            if not buffer:
                continue
            texts = [" ".join(buffer)]
            buffer.clear()
            for idx, result in enumerate(predict_texts(model, texts), start=1):
                print(f"\nBerita {idx}:")
                print(f"  Label         : {result['label']}")
                print(f"  Probabilitas  : {result['probability_valid']:.4f} (ASLI)")
                print(f"  Cuplikan teks : {result['original'][:120]}{'...' if len(result['original']) > 120 else ''}")
            print()
        else:
            buffer.append(line)


def cmd_train(args: argparse.Namespace) -> None:
    print(f"Memuat dataset dari {BASE_DIR.resolve()} ...")
    data = load_dataset(BASE_DIR)
    print(f"Total data setelah pembersihan: {len(data)} sampel.")

    result = train_model(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\n=== Laporan Evaluasi ===")
    print(result.report)
    print("Confusion matrix [HOAX, ASLI]:")
    print(result.confusion)
    print(f"Akurasi: {result.accuracy:.4f} (n={result.test_samples})")

    if not args.no_save:
        save_model(result.model, args.model_path)
        print(f"\nModel tersimpan di {args.model_path.resolve()}")

    if args.interactive:
        interactive_shell(result.model)


def cmd_predict(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    texts: List[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        if not args.file.exists():
            raise SystemExit(f"File {args.file} tidak ditemukan.")
        texts.extend([line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()])

    if not texts:
        print("Masukkan teks melalui STDIN (akhiri dengan CTRL+D / CTRL+Z).")
        try:
            input_text = sys.stdin.read().strip()
        except KeyboardInterrupt:
            return
        if not input_text:
            print("Tidak ada teks yang diberikan.")
            return
        texts = [input_text]

    results = predict_texts(model, texts)
    for idx, result in enumerate(results, start=1):
        print(f"\nBerita {idx}:")
        print(f"  Label         : {result['label']}")
        print(f"  Probabilitas  : {result['probability_valid']:.4f} (ASLI)")
        print(f"  Teks asli     : {result['original']}")
        print(f"  Teks bersih   : {result['cleaned']}")

    if args.interactive:
        interactive_shell(model)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deteksi berita hoaks menggunakan TF-IDF + Logistic Regression.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Latih model dari dataset lokal.")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Proporsi data untuk evaluasi (default: 0.2).")
    train_parser.add_argument("--random-state", type=int, default=42, help="Seed untuk pembagian data.")
    train_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Lokasi penyimpanan model terlatih.")
    train_parser.add_argument("--no-save", action="store_true", help="Jangan simpan model setelah training.")
    train_parser.add_argument("--interactive", action="store_true", help="Masuk ke mode interaktif setelah training.")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Gunakan model terlatih untuk prediksi.")
    predict_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Lokasi model yang akan digunakan.")
    predict_parser.add_argument("--text", nargs="+", help="Satu atau lebih teks untuk diprediksi.")
    predict_parser.add_argument("--file", type=Path, help="File teks (UTF-8) berisi berita, satu per baris.")
    predict_parser.add_argument("--interactive", action="store_true", help="Masuk ke mode interaktif setelah prediksi.")
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.command:
        args = parser.parse_args(["train"])
    args.func(args)


if __name__ == "__main__":
    main()
