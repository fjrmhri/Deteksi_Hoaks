# Deteksi Hoaks (IndoBERT + FastAPI + Frontend)

Sistem ini mendeteksi berita hoaks berbahasa Indonesia menggunakan model IndoBERT
yang dilatih di Google Colab, disajikan melalui FastAPI, dan diakses lewat
frontend web ringan.

## Arsitektur singkat

- **notebooks/**: `indobert_colab_pipeline.py` berisi pipeline Colab untuk
  instalasi dependensi, pemuatan dataset `dataset/Cleaned`, pelatihan, evaluasi,
  serta penyimpanan model/tokenizer ke `models/indobert_hoax/`.
- **backend/**: Layanan FastAPI yang memuat model hasil pelatihan tanpa perlu
  retrain dan mengekspor endpoint prediksi.
- **frontend/**: Halaman web statis sederhana untuk mengirim teks ke backend dan
  menampilkan hasil prediksi beserta skor.

## Jalur penggunaan end-to-end

### 1) Kloning repositori
```bash
git clone https://github.com/fjrmhri/Deteksi_Hoaks.git
cd Deteksi_Hoaks
```

### 2) Latih model di Google Colab
1. Buka `notebooks/indobert_colab_pipeline.py` di Colab (GPU T4).
2. Jalankan sel instalasi dependensi (pinned untuk Colab).
3. Pastikan folder `dataset/Cleaned` ada di Colab dan berisi file Excel dengan
   kolom `hoax` serta kolom teks (mis. `Narasi`, `teks`, dll.).
4. Jalankan sel pipeline untuk: memuat data, split train/validasi, balancing
   sederhana, fine-tuning IndoBERT, evaluasi, dan penyimpanan model.
5. Hasil model/tokenizer akan tersimpan ke `models/indobert_hoax/`.

### 3) Menjalankan backend (FastAPI)
1. (Opsional) buat virtual environment Python.
2. Instal dependensi backend:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Pastikan folder `models/indobert_hoax/` dari Colab tersedia di root repo.
   Jika model berada di lokasi lain, set variabel lingkungan:
   ```bash
   export MODEL_NAME_OR_PATH=/path/ke/model
   ```
4. Jalankan server:
   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. Endpoint utama:
   - `GET /status` untuk health-check.
   - `POST /predict-hoax` (alias `/prediksi` atau `/predict`) dengan payload
     `{ "teks": "isi berita" }` → response `{ label, skor }`.

### 4) Menjalankan frontend
1. Buka folder `frontend/` dengan Live Server (VS Code) atau host statik lain.
2. Isi URL backend (mis. `http://localhost:8000`).
3. Tempel teks berita dan klik **Periksa sekarang** → hasil dan skor tampil di
   panel hasil.

### 5) Alur ringkas model → API → frontend
1. Jalankan notebook → simpan model/tokenizer ke `models/indobert_hoax/`.
2. Backend memuat model itu saat start-up (tidak perlu retrain).
3. Frontend memanggil `POST /predict-hoax` dan menampilkan label + skor.

## Contoh permintaan API
```bash
curl -X POST http://localhost:8000/predict-hoax \ 
  -H "Content-Type: application/json" \ 
  -d '{"teks": "Kabar vaksin bikin tubuh jadi magnet"}'
```

Respons contoh:
```json
{
  "hasil": [
    {"teks": "Kabar vaksin bikin tubuh jadi magnet", "label": "hoax", "skor": 0.98}
  ]
}
```

## Catatan
- Path dataset wajib `dataset/Cleaned` (tidak diubah). Pastikan file Excel ada di
  sana sebelum menjalankan notebook.
- Konfigurasi backend dapat diatur lewat variabel lingkungan: `MODEL_NAME_OR_PATH`,
  `DEVICE`, `MAX_LENGTH`.
- UI sengaja dibuat minimal agar mudah dipakai dan diintegrasikan.
