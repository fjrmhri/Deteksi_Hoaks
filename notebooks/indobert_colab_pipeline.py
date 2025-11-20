# %% [markdown]
"""
IndoBERT hoax detection pipeline for Google Colab (GPU T4 friendly).

- Install dependencies (pinned for stability on Colab free tier).
- Load dataset from **dataset/Cleaned** (expects Excel files with `hoax` label column).
- Clean text, split train/validation, optionally rebalance.
- Fine-tune IndoBERT using Hugging Face `Trainer` with sensible defaults for T4.
- Evaluate the model and save it to `models/indobert_hoax/`.
- Provide a minimal FastAPI example for in-notebook inference.

Use the `# %%` markers as cell separators when opening in VS Code or Colab.
"""

# %%
# Install dependencies (run once per Colab session)
import sys
import subprocess

REQUIRED_PACKAGES = [
    "numpy==1.26.4",
    "torch==2.2.2",
    "transformers==4.44.2",
    "datasets==2.18.0",
    "accelerate==0.34.2",
    "pandas==2.2.2",
    "scikit-learn==1.4.2",
    "fastapi==0.110.2",
    "uvicorn==0.29.0",
]

if "google.colab" in sys.modules:  # Only install when running in Colab
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + REQUIRED_PACKAGES)

# %%
# Imports and configuration
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class TrainingConfig:
    dataset_dir: Path = Path("dataset/Cleaned")
    model_name: str = "indolem/indobert-base-uncased"
    max_length: int = 256
    train_batch_size: int = 8  # safe for T4 with grad accumulation
    eval_batch_size: int = 32
    grad_accumulation: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    output_dir: Path = Path("outputs/indobert_hoax")
    save_dir: Path = Path("models/indobert_hoax")
    seed: int = 42
    balance_minority: bool = True  # simple duplication to balance classes

cfg = TrainingConfig()
set_seed(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# %%
# Data loading utilities

def _pick_text_column(df: pd.DataFrame) -> str:
    candidates = ["Clean Narasi", "Narasi", "isi_berita", "teks", "text", "isi", "artikel", "judul"]
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def load_dataset_from_excels(dataset_dir: Path) -> pd.DataFrame:
    """Load and combine all Excel files inside dataset/Cleaned."""
    if not dataset_dir.exists():
        raise FileNotFoundError("dataset/Cleaned must exist and contain Excel files.")

    excel_files = sorted(dataset_dir.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError("No Excel files found in dataset/Cleaned.")

    frames: List[pd.DataFrame] = []
    for path in excel_files:
        df = pd.read_excel(path)
        if "hoax" not in df.columns:
            raise ValueError(f"File {path.name} is missing 'hoax' column.")
        text_col = _pick_text_column(df)
        subset = df[[text_col, "hoax"]].copy()
        subset.columns = ["text", "label"]
        subset["source"] = path.stem
        frames.append(subset)

    data = pd.concat(frames, ignore_index=True)
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(int)
    data = data.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    return data


# %%
# Preprocessing and splitting
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Basic normalisation mirroring inference time."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def prepare_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]

    # Simple balancing by duplicating minority examples
    if cfg.balance_minority:
        counts = df["label"].value_counts()
        min_label = counts.idxmin()
        max_count = counts.max()
        minority_df = df[df["label"] == min_label]
        dup_count = max_count - len(minority_df)
        if dup_count > 0:
            extra = minority_df.sample(dup_count, replace=True, random_state=cfg.seed)
            df = pd.concat([df, extra], ignore_index=True)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=cfg.seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# %%
# Tokenisation helpers

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)


def build_dataset(df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(df)

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=cfg.max_length,
        )

    return ds.map(tokenize, batched=True, remove_columns=["text", "__index_level_0__"] if "__index_level_0__" in ds.column_names else ["text"])


# %%
# Metrics

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


# %%
# Training

def train_model(train_ds: Dataset, val_ds: Dataset) -> Trainer:
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label={0: "not_hoax", 1: "hoax"},
        label2id={"not_hoax": 0, "hoax": 1},
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    trainer.train()
    return trainer


# %%
# Evaluation and saving

def evaluate_and_save(trainer: Trainer, val_ds: Dataset) -> Dict[str, float]:
    metrics = trainer.evaluate(val_ds)
    print("Validation metrics:", metrics)

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"Model and tokenizer saved to {cfg.save_dir}")
    return metrics


# %%
# Run the full pipeline
if __name__ == "__main__":
    raw_df = load_dataset_from_excels(cfg.dataset_dir)
    train_df, val_df = prepare_splits(raw_df)
    train_ds = build_dataset(train_df)
    val_ds = build_dataset(val_df)

    trainer = train_model(train_ds, val_ds)
    evaluate_and_save(trainer, val_ds)

# %%
# Inference helper for testing inside the notebook (after training)

def predict_texts(texts: List[str], model_path: Path = cfg.save_dir) -> List[Dict[str, float]]:
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tok = AutoTokenizer.from_pretrained(model_path)
    encoded = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_length).to(device)
    with torch.inference_mode():
        logits = model(**encoded).logits
        probs = F.softmax(logits, dim=-1)
        scores, ids = probs.max(dim=-1)
    id2label = {int(k): v for k, v in model.config.id2label.items()} if model.config.id2label else {0: "not_hoax", 1: "hoax"}
    return [
        {"label": id2label.get(i.item(), str(i.item())), "score": round(s.item(), 4)}
        for s, i in zip(scores, ids)
    ]


# %% [markdown]
"""
## FastAPI mini example (in-notebook)

Run after training/saving to quickly test inference without leaving the notebook.

The server enables permissive CORS to allow requests from hosted frontends (e.g., Vercel).

> ⚠️ Jika frontend kamu berjalan di `https://` (seperti Vercel), browser akan menolak
> permintaan ke backend `http://` biasa (mixed content). Pastikan backend juga
> tersedia lewat `https` (mis. reverse proxy/Cloudflare Tunnel) atau uji dari
> halaman yang juga memakai `http://`.
"""

# %%
import socket
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="IndoBERT Hoax Detection")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_for_api = AutoModelForSequenceClassification.from_pretrained(cfg.save_dir).to(device)
tokenizer_for_api = AutoTokenizer.from_pretrained(cfg.save_dir)
label_map = {int(k): v for k, v in model_for_api.config.id2label.items()} if model_for_api.config.id2label else {0: "not_hoax", 1: "hoax"}


class PredictPayload(BaseModel):
    """Accept both English (`text`) and Indonesian (`teks`) keys."""

    text: str | None = Field(default=None, description="Teks berita untuk dianalisis")
    teks: str | None = Field(default=None, description="Alias Bahasa Indonesia untuk 'text'")

    def resolve_text(self) -> str:
        if self.text:
            return self.text
        if self.teks:
            return self.teks
        raise HTTPException(status_code=422, detail="Field 'text' atau 'teks' wajib diisi.")


class PredictResponse(BaseModel):
    label: str
    score: float


@app.post("/predict-hoax", response_model=PredictResponse)
def predict_endpoint(payload: PredictPayload) -> Any:
    encoded = tokenizer_for_api(
        payload.resolve_text(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
    ).to(device)
    with torch.inference_mode():
        logits = model_for_api(**encoded).logits
        prob = F.softmax(logits, dim=-1)[0]
        score, idx = prob.max(dim=-1)
    return {
        "label": label_map.get(idx.item(), str(idx.item())),
        "score": round(score.item(), 4),
    }

def resolve_public_base_url(port: int = 8000) -> str:
    """Return a base URL that can be used by an already-hosted website.

    On most hosted environments (VPS/cloud/Colab with port forwarding) the external
    IP can be retrieved via https://ifconfig.me. If the request fails, fall back to
    the machine hostname. The result is meant to be plugged into the frontend as
    `http://<ip>:<port>`.
    """

    try:
        external_ip = requests.get("https://ifconfig.me", timeout=5).text.strip()
    except Exception:
        external_ip = socket.gethostbyname(socket.gethostname())
    return f"http://{external_ip}:{port}"


# To run inside Colab/hosted notebook and print a public-ish URL, uncomment:
# import nest_asyncio, uvicorn
# nest_asyncio.apply()
# port = 8000
# print("Base URL for frontend:", resolve_public_base_url(port))
# uvicorn.run(app, host="0.0.0.0", port=port)

# For a quick smoke test without starting a server:
if __name__ == "__main__":
    sample = "Vaksin bikin tubuh jadi magnet adalah kabar bohong."  # type: ignore[unreachable]
    print(predict_texts([sample]))
