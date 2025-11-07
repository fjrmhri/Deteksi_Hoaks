# Deteksi Hoaks Real-Time

Repositori ini menambahkan layanan **FastAPI** untuk menjalankan model IndoBERT
sebagai detektor hoaks serta antarmuka web statis untuk mengonsumsi API secara
real-time. Arsitektur ditujukan untuk dijalankan pada Google Colab menggunakan
GPU T4 dan dipublikasikan melalui **ngrok**.

## Struktur Proyek

```
app/
  config.py        # Konfigurasi layanan FastAPI (model, device, panjang maksimum)
  inference.py     # Pemanggilan model IndoBERT untuk inferensi
  main.py          # Endpoint FastAPI `/status` dan `/prediksi`
  text_utils.py    # Normalisasi teks sederhana
frontend/
  index.html       # Halaman web untuk memasukkan teks berita
  styles.css       # Styling dengan dark mode modern
  app.js           # Logika fetch ke endpoint FastAPI (via ngrok)
README.md          # Panduan lengkap integrasi end-to-end
requirements.txt   # Dependensi Python
```

## Menjalankan Layanan di Google Colab

1. **Salin repositori** ke Google Drive atau unduh langsung di notebook Colab.
2. **Instal dependensi** utama:

   ```python
   !pip install -r requirements.txt
   !pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
   !pip install transformers fastapi uvicorn[standard] pydantic python-multipart
   !pip install pyngrok
   ```

3. **Siapkan model IndoBERT** yang sudah dilatih untuk klasifikasi hoaks. Salin
   folder model (misal `model_terbaik/`) sehingga berisi `config.json`,
   `pytorch_model.bin`, dan `tokenizer.json`.

4. **Jalankan server FastAPI** di sel Colab:

   ```python
   import nest_asyncio
   import uvicorn
   from pyngrok import ngrok
   from app.config import get_settings

   # Opsional: setel lokasi model jika tidak berada di `model_terbaik`
   import os
   os.environ["MODEL_NAME_OR_PATH"] = "/content/model_terbaik"

   # Buka tunnel ngrok ke port 8000
   public_url = ngrok.connect(8000, bind_tls=True).public_url
   print("Public URL:", public_url)

   # Jalankan uvicorn (wajib memanggil nest_asyncio untuk Colab)
   nest_asyncio.apply()
   uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
   ```

   Endpoint `POST {public_url}/prediksi` kini tersedia untuk menerima payload:

   ```json
   { "teks": "Isi berita yang ingin diperiksa" }
   ```

   Respons contoh:

   ```json
   {
     "hasil": [
       {
         "teks": "Isi berita yang diperiksa",
         "label": "hoaks",
         "skor": 0.9821
       }
     ]
   }
   ```

5. **Gunakan frontend** dengan membuka `frontend/index.html` (misal melalui
   GitHub Pages, Vercel, atau Live Server VS Code). Masukkan URL ngrok pada
   kolom “Alamat API (ngrok)” lalu tekan “Periksa Sekarang”.

## Opsi Kustomisasi

- `MODEL_NAME_OR_PATH`: Tentukan path model berbeda melalui variabel lingkungan.
- `DEVICE`: Atur menjadi `cpu` untuk memaksa CPU.
- `MAX_LENGTH`: Sesuaikan panjang tokenisasi (default 256).

## Kesehatan Layanan

Gunakan endpoint `GET /status` untuk mengecek konfigurasi aktif dan memastikan
server responsif.

## Lisensi

Gunakan sesuai kebutuhan penelitian atau pengembangan internal.
