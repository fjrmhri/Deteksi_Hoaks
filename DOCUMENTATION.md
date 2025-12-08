# Dokumentasi Sistem Deteksi Hoaks (IndoBERT)

Dokumentasi ini menjelaskan alur end-to-end sistem deteksi hoaks: notebook pelatihan, backend FastAPI di Hugging Face Spaces, dan frontend statis di folder `public`. Seluruh penjelasan menggunakan Bahasa Indonesia.

## Ringkasan Singkat

- **Model**: `indolem/indobert-base-uncased` yang di-fine-tune menjadi klasifikator biner `not_hoax` vs `hoax`.
- **Dataset**: Gabungan CNN, Detik, Kompas, TurnBackHoax, dan korpus non-hoaks 2020+ (177k baris, deduplikasi, imputasi label, oversampling hoaks di train).
- **Inferensi API**: FastAPI `/predict` & `/predict-batch`, payload JSON, mengembalikan probabilitas, label, serta level risiko (low/medium/high).
- **Frontend**: Halaman statis (`public`) yang memanggil API Space Hugging Face melalui `fetch`, menampilkan badge risiko dan penjelasan.

---

## 1. Notebook `notebooks/Deteksi_Hoax.ipynb`

### Dataset & Alur Persiapan

- **Sumber data**: `Summarized_CNN.csv`, `Summarized_Detik.csv`, `Summarized_Kompas.csv`, `Summarized_TurnBackHoax.csv`, `Summarized_2020+.csv` (diunduh via `kagglehub`).
- **Normalisasi kolom**: Dipastikan kolom standar `url, judul, tanggal, isi_berita, Narasi, Clean Narasi, hoax, summary, source`.
- **Ekstraksi teks**: Fungsi `pick_text` memilih prioritas `Clean Narasi` → `Narasi` → `isi_berita` → `judul`.
- **Labeling**:
  - Jika label kosong untuk sumber `cnn/detik/kompas/merged_extra` → dianggap `not_hoax (0)`.
  - Jika sumber `turnbackhoax` → dianggap `hoax (1)` bila kosong.
  - Baris kosong & label NaN dibuang; duplikat `(text, label)` dibuang.
- **Distribusi awal**: ~164k non-hoax vs 11.9k hoax (imbalanced).
- **Split**: Stratified 70% train / 15% val / 15% test.
- **Balancing**: Oversampling minoritas (kelas hoax) hanya pada set train hingga seimbang.

### Tokenisasi & Pipeline

- Tokenizer: `AutoTokenizer.from_pretrained(indolem/indobert-base-uncased)`.
- `max_length=256`, truncation + padding dinamis (`DataCollatorWithPadding`).
- Dataset HuggingFace (`Dataset.from_pandas`) dengan format tensor `input_ids, attention_mask, label`.

### Pelatihan

- Model: `AutoModelForSequenceClassification(num_labels=2, id2label={"not_hoax":0,"hoax":1})`.
- Hyperparameter utama: lr `2e-5`, weight decay `0.01`, epochs `3`, batch size train `96` (grad accumulation 2), eval `384`, fp16 jika GPU.
- Trainer + `compute_metrics`: accuracy, precision/recall/F1 (binary & weighted).

### Evaluasi

- **Validation**: Accuracy 0.9983, F1 hoax 0.9877, confusion matrix menunjukkan FP rendah (14) & FN 30.
- **Test**: Accuracy 0.9983, F1 hoax 0.9874, confusion matrix [[24630,11],[34,1759]].
- **Interpretasi**: Model sangat sensitif untuk hoaks; tetap ada potensi FN pada teks pendek/noisy.

### Penyimpanan Model

- Disimpan ke folder `indobert_hoax_model_v3` lalu di-zip untuk diunggah ke Hugging Face Hub/Space.

---

## 2. Backend (Hugging Face Space) — `backend/app.py`

### Struktur & Dependensi

- FastAPI + Pydantic schema: `PredictRequest`, `BatchPredictRequest`, `PredictResponse`, `BatchPredictResponse`.
- Model & tokenizer di-load dari Hub: `MODEL_ID` (default `fjrmhri/hoaks-detection`), `MODEL_SUBFOLDER` (default `models/indobert_hoax`).
- Konfigurasi env:
  - `MAX_LENGTH` (default 256)
  - `HOAX_THRESH_HIGH` (0.98), `HOAX_THRESH_MED` (0.60)
  - Logging: `ENABLE_HOAX_LOGGING`, `HOAX_LOG_SAMPLE_RATE`
- Device otomatis CUDA bila tersedia.

### Alur Inferensi

1. **Preprocess**: `_prepare_texts` memastikan teks non-kosong, mengganti kosong dengan placeholder `[EMPTY]`.
2. **Tokenisasi**: padding+truncation, `max_length`.
3. **Model forward**: `model.eval()`, `torch.no_grad()`, softmax → probabilitas.
4. **Risk analysis**: `analyze_risk`:
   - `p_hoax > 0.98` → `high`
   - `0.60 < p_hoax ≤ 0.98` → `medium`
   - `≤ 0.60` → `low`
   - Teks < 5 kata otomatis minimal `medium`.
5. **Logging sampling** opsional (sampling rate).

### Endpoint

- `GET /` info model & threshold.
- `GET /health` health check.
- `POST /predict`
  - Body: `{ "text": "<string>" }`
  - Respons: `label`, `score` (prob tertinggi), `probabilities`, `hoax_probability`, `risk_level`, `risk_explanation`.
- `POST /predict-batch`
  - Body: `{ "texts": ["t1","t2", ...] }`
  - Respons: `{ results: [PredictResponse, ...] }`

### Komunikasi Frontend ↔ Backend

- CORS diizinkan untuk semua origin.
- Frontend memakai fetch POST JSON; tidak ada auth di versi ini.

---

## 3. Frontend — `public/`

### File Utama

- `index.html`: form textarea, status panel, hasil prediksi dengan badge risiko, info model statis.
- `app.js`: logika UI & API call.
- `styles.css`: styling (tidak diubah pada tugas ini).

### Alur UI → API

1. Pengguna menempel teks berita, klik **Periksa sekarang** (atau Ctrl+Enter).
2. `app.js` memanggil `callApi` → POST `${API_BASE}/predict` dengan JSON `{text}`.
3. Jika sukses:
   - `extractPrediction` memastikan format.
   - `renderResult` menampilkan badge risiko, keputusan (hoaks/bukan), skor/probabilitas, penjelasan, dan teks asli.
4. Tombol **Copy** / **Share** menyiapkan ringkasan hasil untuk clipboard atau Web Share API.
5. Inisialisasi awal memanggil `/health`; status koneksi ditampilkan di panel status.

### Payload & Respons

- **Request**: `POST /predict` headers `Content-Type: application/json`, body `{ "text": "..." }`.
- **Response** (contoh):
  ```json
  {
    "label": "hoax",
    "score": 0.99,
    "probabilities": { "not_hoax": 0.01, "hoax": 0.99 },
    "hoax_probability": 0.99,
    "risk_level": "high",
    "risk_explanation": "Model sangat yakin..."
  }
  ```

---

## 4. Diagram (teks / ASCII)

### 4.1 System Architecture

```
[User Browser]
    |
    v
[Frontend JS (public/app.js)]
    | POST /predict
    v
[FastAPI on HF Space]
    | -> Tokenizer (IndoBERT)
    | -> IndoBERT Classifier (PT)
    | <- Probabilities + Risk
    v
[Response JSON]
```

### 4.2 Data Flow Diagram (DFD Level-1)

```
Dataset CSV (CNN/Detik/Kompas/TBH/2020+)
    | load_all_datasets()
    v
Preprocess & Labeling (pick_text, imputasi label, drop kosong/duplikat)
    | stratified_splits() 70/15/15
    | balance_minority_only_train()
    v
Tokenisasi (AutoTokenizer, max_len=256)
    v
Training (Trainer, IndoBERT) --> Model terlatih
    v
Hugging Face Hub / Space (MODEL_ID + SUBFOLDER)
    v
API FastAPI (/predict, /predict-batch)
    v
Frontend fetch & render
```

### 4.3 Activity / Process Flow (inferensi)

```
User input teks -> Klik submit
    -> Frontend kirim POST /predict
        -> Backend tokenisasi
        -> Model forward + softmax
        -> Hitung p_hoax & risk_level
        -> Response JSON
    -> Frontend render badge + penjelasan
```

### 4.4 UML Sequence (ringkas)

```
User -> Browser: ketik & submit
Browser -> FastAPI: POST /predict {text}
FastAPI -> Tokenizer: encode(text, max_len=256)
Tokenizer -> FastAPI: input_ids, attention_mask
FastAPI -> Model: forward(...)
Model -> FastAPI: logits -> softmax -> prob
FastAPI -> Browser: {label, prob, risk_level, explanation}
Browser -> User: tampilkan hasil
```

### 4.5 UML Class Diagram (backend utama)

```
+------------------+
| FastAPI app      |
+------------------+
| - tokenizer      |
| - model          |
| - ID2LABEL       |
| - THRESH_*       |
+------------------+
| +/predict()      |
| +/predict-batch()|
| +/health()       |
+------------------+
        ^
        |
+------------------+
| Schemas (Pydantic)|
+------------------+
| PredictRequest    |
| BatchPredictRequest|
| PredictResponse   |
| BatchPredictResponse|
+------------------+
```

---

## 5. Alur Sistem End-to-End

1. **Training** di notebook (Colab): muat dataset → preprocessing & balancing → tokenisasi → fine-tune IndoBERT → simpan model.
2. **Deployment**: model diunggah ke Hugging Face Hub/Space; FastAPI memuat `MODEL_ID` + `SUBFOLDER`, siap melayani inferensi.
3. **Inferensi**: frontend statis memanggil API `/predict` atau `/predict-batch`; backend mengembalikan skor & level risiko; UI menampilkan hasil, menyediakan copy/share.
4. **Risk tuning**: ambang risiko dapat diatur lewat env (`HOAX_THRESH_HIGH/MED`); teks pendek otomatis dinaikkan minimal `medium`.

---

## 6. Catatan Implementasi & Operasional

- Gunakan HTTPS saat memanggil Space untuk menghindari blok CORS.
- Space default tanpa autentikasi; jika dipublikasi luas, pertimbangkan rate-limit atau key sederhana.
- Logging sampel dapat diaktifkan via env untuk audit (`ENABLE_HOAX_LOGGING=1`).
- Model sensitif terhadap konteks; teks <5 kata diberi peringatan karena ketidakstabilan prediksi.
- Untuk reproduksi training, perlukan kredensial Kaggle (datasets) dan token Hugging Face (login).

---

## 7. Checklist Uji Cepat

- `/health` mengembalikan `{status: "ok"}`.
- `/predict` dengan teks valid mengembalikan `label`, `probabilities`, dan `risk_level`.
- Frontend menampilkan badge risiko dan penjelasan; tombol Copy/Share berfungsi.
- Threshold risiko sesuai env (high >0.98, medium >0.60).
