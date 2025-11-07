# Hoax Detection Website (IndoBERT + FastAPI)

Repositori ini disederhanakan untuk fokus pada pengembangan website deteksi
hoaks yang memakai model IndoBERT melalui layanan **FastAPI** dan alur kerja
Google Colab. Semua artefak yang tidak terkait langsung dengan aplikasi web
(datasets mentah, skrip eksperimen lama, dan cache Python) telah dipindahkan
atau dihapus agar struktur proyek tetap bersih dan mudah dipahami.

## Struktur Proyek

```
backend/
  app/                # Kode layanan FastAPI yang mengekspor IndoBERT
    __init__.py
    config.py
    inference.py
    main.py
    text_utils.py
  requirements.txt    # Dependensi Python untuk backend
frontend/
  index.html          # Halaman web utama (statik)
  styles.css          # Styling antarmuka
  app.js              # Logika interaksi dengan API FastAPI
notebooks/
  indobert_colab_pipeline.py  # Skrip utilitas untuk dieksekusi di Google Colab
README.md
.gitignore
```

## Backend FastAPI

### Menjalankan Secara Lokal

1. Buat dan aktifkan virtual environment Python (opsional namun disarankan).
2. Instal dependensi backend:

   ```bash
   pip install -r backend/requirements.txt
   ```

3. Pastikan folder model hasil fine-tuning IndoBERT tersedia (contoh:
   `model_terbaik/` berisi `config.json`, `pytorch_model.bin`, `tokenizer.json`,
   dan artefak tokenizer lainnya).
4. Jalankan layanan FastAPI menggunakan Uvicorn:

   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
   ```

5. Endpoint utama:
   - `GET /status` untuk health-check.
   - `POST /prediksi` dengan payload `{ "teks": "..." }` untuk memperoleh
     label `hoaks` atau `bukan hoaks` beserta skor probabilitas.

Konfigurasi runtime (lokasi model, perangkat, panjang maksimum token) dapat
pengguna atur melalui variabel lingkungan: `MODEL_NAME_OR_PATH`, `DEVICE`, dan
`MAX_LENGTH`. File `.env` (jika ada) juga akan dibaca secara otomatis oleh
`backend/app/config.py`.

### Menjalankan di Google Colab

1. Unggah repositori ini dan folder model IndoBERT ke sesi Colab.
2. Instal dependensi backend:

   ```python
   !pip install -r backend/requirements.txt
   !pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
   !pip install transformers uvicorn[standard] pyngrok nest_asyncio
   ```

3. Set variabel lingkungan bila model tidak berada di `./model_terbaik`:

   ```python
   import os
   os.environ["MODEL_NAME_OR_PATH"] = "/content/drive/MyDrive/model_anda"
   ```

4. Buka tunnel ngrok dan jalankan Uvicorn:

   ```python
   import nest_asyncio
   from pyngrok import ngrok
   import uvicorn

   public_url = ngrok.connect(8000, bind_tls=True).public_url
   print("Public URL:", public_url)

   nest_asyncio.apply()
   uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000)
   ```

URL ngrok inilah yang dipakai frontend untuk melakukan prediksi secara real-time
melalui endpoint `/prediksi`.

## Frontend Statis

`frontend/` berisi aset HTML, CSS, dan JavaScript sederhana. Jalankan halaman
tersebut menggunakan fitur *Live Server* di VS Code atau host statik lainnya,
kemudian masukkan URL ngrok (atau URL backend lokal) ke kolom "Alamat API".
Klik **Periksa Sekarang** untuk mengirim teks berita dan menerima hasil
klasifikasi.

## Pipeline di Google Colab

Skrip `notebooks/indobert_colab_pipeline.py` adalah versi pythonic dari notebook
Colab yang memuat pipeline lengkap: persiapan dependensi, pemuatan dataset,
augmentasi, fine-tuning IndoBERT, evaluasi, penyimpanan model, hingga contoh API
FastAPI in-notebook. Dataset tidak disertakan dalam repositori — unggah dataset
sendiri ke Colab (misalnya ke `/content/dataset/`) sebelum mengeksekusi pipeline.

## Catatan Tambahan

- Folder atau file keluaran seperti `model_terbaik/`, `.env`, dan dataset lokal
  sudah masuk ke `.gitignore` agar tidak ikut terlacak.
- Struktur baru ini memisahkan backend, frontend, dan utilitas Colab sehingga
  workflow deployment menjadi lebih jelas dan terfokus pada website deteksi
  hoaks.
