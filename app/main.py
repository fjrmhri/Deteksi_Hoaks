"""FastAPI application exposing IndoBERT hoax detection via HTTP."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .inference import get_detector

app = FastAPI(
    title="Deteksi Hoaks Indonesia",
    description=(
        "API FastAPI untuk mendeteksi berita hoaks menggunakan model IndoBERT "
        "yang dilatih khusus untuk klasifikasi hoaks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    teks: str = Field(..., description="Konten berita yang ingin diperiksa.")


class PredictionItem(BaseModel):
    teks: str = Field(..., description="Teks asli yang diprediksi.")
    label: str = Field(..., description='Label prediksi: "hoaks" atau "bukan hoaks".')
    skor: float = Field(..., ge=0.0, le=1.0, description="Skor probabilitas untuk label terpilih.")


class PredictionResponse(BaseModel):
    hasil: List[PredictionItem]


@app.get("/status")
def status() -> Dict[str, Any]:
    """Health-check endpoint for monitoring."""

    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.model_name_or_path,
        "device": settings.device,
        "max_length": settings.max_length,
    }


@app.post("/prediksi", response_model=PredictionResponse)
def prediksi(payload: PredictionRequest) -> PredictionResponse:
    """Return hoax predictions for the supplied text."""

    teks = payload.teks.strip()
    if not teks:
        raise HTTPException(status_code=422, detail="Teks tidak boleh kosong.")

    detector = get_detector()
    label, score = detector.predict(teks)
    item = PredictionItem(teks=teks, label=label, skor=round(score, 4))
    return PredictionResponse(hasil=[item])
