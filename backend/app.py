import os
import random
from typing import List, Dict, Tuple, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Konfigurasi & Load Model
# =========================

MODEL_ID = os.getenv("MODEL_ID", "fjrmhri/hoaks-detection")
SUBFOLDER = os.getenv("MODEL_SUBFOLDER", "models/indobert_hoax") or None
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

# Threshold risiko (bisa diubah lewat env)
THRESH_HIGH = float(os.getenv("HOAX_THRESH_HIGH", "0.98"))
THRESH_MED = float(os.getenv("HOAX_THRESH_MED", "0.60"))

# Logging sampling
ENABLE_LOGGING = os.getenv("ENABLE_HOAX_LOGGING", "0") == "1"
LOG_SAMPLE_RATE = float(os.getenv("HOAX_LOG_SAMPLE_RATE", "0.2"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("======================================")
print(f"Loading model from Hub: {MODEL_ID}")
print(f"Using subfolder: {SUBFOLDER}")
print(f"Running on device: {DEVICE}")
print(f"THRESH_HIGH = {THRESH_HIGH}, THRESH_MED = {THRESH_MED}")
print(f"ENABLE_LOGGING = {ENABLE_LOGGING}, LOG_SAMPLE_RATE = {LOG_SAMPLE_RATE}")
print("======================================")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder=SUBFOLDER)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, subfolder=SUBFOLDER
)
model.to(DEVICE)
model.eval()

# Mapping id → label
if getattr(model.config, "id2label", None):
    ID2LABEL: Dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
else:
    ID2LABEL = {0: "not_hoax", 1: "hoax"}


# =========================
# FastAPI setup
# =========================

app = FastAPI(
    title="Indo Hoax Detector API",
    description="API FastAPI untuk deteksi berita hoaks (model IndoBERT).",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # bisa dibatasi ke domain Vercel jika perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Schemas
# =========================

class PredictRequest(BaseModel):
    text: str


class BatchPredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    label: str
    score: float
    probabilities: Dict[str, float]
    hoax_probability: float
    risk_level: str
    risk_explanation: str


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


# =========================
# Util inferensi
# =========================

def _prepare_texts(texts: List[str]) -> List[str]:
    processed = []
    for t in texts:
        if t is None:
            t = ""
        t = str(t).strip()
        if t == "":
            processed.append("[EMPTY]")
        else:
            processed.append(t)
    return processed


def _predict_proba(texts: List[str]) -> List[Dict[str, float]]:
    if not texts:
        return []

    texts = _prepare_texts(texts)

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results: List[Dict[str, float]] = []
    for row in probs:
        prob_dict: Dict[str, float] = {}
        for idx, p in enumerate(row):
            label_name = ID2LABEL.get(idx, str(idx))
            prob_dict[label_name] = float(p)
        results.append(prob_dict)

    return results


def _extract_hoax_probability(prob_dict: Dict[str, float]) -> float:
    # kasus ideal: label "hoax"
    if "hoax" in prob_dict:
        return float(prob_dict["hoax"])

    # fallback: key yang mengandung "hoax"
    for k, v in prob_dict.items():
        if "hoax" in k.lower():
            return float(v)

    # fallback: kalau cuma 2 label dan ada "not_hoax"
    if len(prob_dict) == 2 and "not_hoax" in prob_dict:
        for k, v in prob_dict.items():
            if k != "not_hoax":
                return float(v)

    return 0.0


def analyze_risk(prob_dict: Dict[str, float], original_text: Optional[str] = None) -> Tuple[float, str, str]:
    """
    Dari prob_dict + teks asli → (p_hoax, risk_level, risk_explanation)
    Threshold:
      - p_hoax > THRESH_HIGH         → high   (Hoaks – tingkat tinggi)
      - THRESH_MED < p_hoax ≤ HIGH   → medium (Perlu dicek / curiga)
      - p_hoax ≤ THRESH_MED          → low    (Cenderung bukan hoaks)
    Teks sangat pendek (< 5 kata) → minimal 'medium'
    """
    p_hoax = _extract_hoax_probability(prob_dict)

    # default berdasarkan probabilitas
    if p_hoax > THRESH_HIGH:
        level = "high"
        explanation = (
            f"Model sangat yakin teks ini hoaks (P(hoaks) ≈ {p_hoax:.2%}). "
            "Sebaiknya jangan dipercaya sebelum ada klarifikasi resmi atau sumber tepercaya."
        )
    elif p_hoax > THRESH_MED:
        level = "medium"
        explanation = (
            f"Model menilai teks ini berpotensi hoaks (P(hoaks) ≈ {p_hoax:.2%}). "
            "Disarankan untuk mengecek ulang ke sumber resmi sebelum menyebarkan."
        )
    else:
        level = "low"
        explanation = (
            f"Model menilai teks ini cenderung bukan hoaks (P(hoaks) ≈ {p_hoax:.2%}). "
            "Meski demikian, tetap gunakan literasi dan bandingkan dengan sumber lain."
        )

    # Perlakuan khusus teks sangat pendek
    if original_text is not None:
        word_count = len(str(original_text).strip().split())
        if word_count < 5:
            # Jangan pernah beri 'low' untuk teks terlalu pendek
            if level == "low":
                level = "medium"
            explanation += (
                " Catatan: teks ini sangat pendek (< 5 kata), sehingga prediksi model "
                "bisa kurang stabil. Gunakan hasil ini dengan ekstra hati-hati."
            )

    return p_hoax, level, explanation


def _maybe_log(sample_info: Dict):
    if not ENABLE_LOGGING:
        return
    if random.random() > LOG_SAMPLE_RATE:
        return
    # Logging simpel ke stdout
    print("[HOAX_LOG]", sample_info)


# =========================
# Routes
# =========================

@app.get("/")
def read_root():
    return {
        "message": "Indo Hoax Detector API is running.",
        "model_id": MODEL_ID,
        "subfolder": SUBFOLDER,
        "labels": ID2LABEL,
        "max_length": MAX_LENGTH,
        "device": str(DEVICE),
        "risk_thresholds": {
            "high": f"P(hoaks) > {THRESH_HIGH}",
            "medium": f"{THRESH_MED} < P(hoaks) ≤ {THRESH_HIGH}",
            "low": f"P(hoaks) ≤ {THRESH_MED}",
        },
        "logging": {
            "enabled": ENABLE_LOGGING,
            "sample_rate": LOG_SAMPLE_RATE,
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    original_text = request.text
    prob_list = _predict_proba([original_text])

    if not prob_list:
        return PredictResponse(
            label="unknown",
            score=0.0,
            probabilities={},
            hoax_probability=0.0,
            risk_level="low",
            risk_explanation="Teks kosong, tidak dapat dievaluasi.",
        )

    prob_dict = prob_list[0]
    label = max(prob_dict, key=prob_dict.get)
    score = prob_dict[label]

    p_hoax, risk_level, risk_explanation = analyze_risk(prob_dict, original_text=original_text)

    _maybe_log({
        "route": "/predict",
        "text_len": len(str(original_text)),
        "word_count": len(str(original_text).split()),
        "label": label,
        "p_hoax": p_hoax,
        "risk_level": risk_level,
    })

    return PredictResponse(
        label=label,
        score=float(score),
        probabilities=prob_dict,
        hoax_probability=float(p_hoax),
        risk_level=risk_level,
        risk_explanation=risk_explanation,
    )


@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    texts = request.texts or []
    prob_list = _predict_proba(texts)
    results: List[PredictResponse] = []

    for original_text, prob_dict in zip(texts, prob_list):
        label = max(prob_dict, key=prob_dict.get)
        score = prob_dict[label]
        p_hoax, risk_level, risk_explanation = analyze_risk(prob_dict, original_text=original_text)

        _maybe_log({
            "route": "/predict-batch",
            "text_len": len(str(original_text)),
            "word_count": len(str(original_text).split()),
            "label": label,
            "p_hoax": p_hoax,
            "risk_level": risk_level,
        })

        results.append(
            PredictResponse(
                label=label,
                score=float(score),
                probabilities=prob_dict,
                hoax_probability=float(p_hoax),
                risk_level=risk_level,
                risk_explanation=risk_explanation,
            )
        )

    return BatchPredictResponse(results=results)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
