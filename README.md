<p align="center">
  <img src="https://img.shields.io/github/stars/fjrmhri/Deteksi_Hoaks?style=for-the-badge&logo=github&color=8b5cf6" alt="Stars"/>
  <img src="https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Transformers-4.57.1-ffcc00?style=for-the-badge&logo=huggingface" alt="Transformers"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
</p>

Deteksi Hoaks Indonesia – IndoBERT Hoax Detector

Aplikasi Deteksi Hoaks Indonesia adalah sistem end-to-end berbasis IndoBERT yang mendeteksi apakah sebuah teks berita mengandung indikasi hoaks atau bukan.
Pipeline meliputi:

Model IndoBERT-base yang di-fine-tune dengan dataset besar hoaks + non-hoaks

Backend FastAPI yang berjalan di Hugging Face Spaces

Frontend statis (HTML/JS/CSS) yang bisa dideploy ke Vercel

Skema risk level berdasarkan probabilitas hoaks: low, medium, high

1. Arsitektur Sistem
User → Frontend (Vercel) → FastAPI (HuggingFace Spaces) → IndoBERT Model → Prediksi


Komponen:

Model: indolem/indobert-base-uncased (fine-tuned)

Endpoint utama: /predict

Output: label, probabilitas, dan risk level

2. Dataset & Pipeline Data
2.1 Dataset yang Digunakan
Non-hoaks:

Summarized_CNN.csv

Summarized_Detik.csv

Summarized_Kompas.csv

merged_clean_filtered_2020plus_halfNaT.csv
(hasil merging + cleaning besar, filtering tanggal ≥ 2020, dan sebagian NaT dibuang)

Hoaks:

Summarized_TurnBackHoax.csv
(berisi narasi hoaks & debunk)

Setelah diselaraskan, skema final dataset:

url
judul
tanggal
isi_berita
Narasi
Clean Narasi
hoax   (0=non-hoaks, 1=hoaks)
summary

3. Cleaning dan Merging Data
3.1 Cleaning teks (Clean Narasi)

Pipeline:

lowercase

hapus HTML tags

hapus URL

hapus emoji, simbol, angka tidak penting

normalize repeated characters ("hebooooh" → "heboh")

stopwords removal

stemming ringan (Bahasa Indonesia)

slang normalization (yg→yang, gk→nggak, dll)

spell correction ringan

trim whitespace

Jika suatu sumber tidak punya kolom tertentu, diisi "".

3.2 Merging

Semua CSV dibaca

Kolom dirapikan

Dedup berdasarkan (url, judul, Clean Narasi)

Buang baris kosong

Buang sebagian NaT (agar dataset lebih kecil & bersih)

4. Filter Tanggal & Ukuran Dataset

Dataset besar difilter:

Semua tanggal < 2020 dibuang

NaT dikurangi 50% secara random

Contoh perubahan ukuran:

Sebelum filter: 196.928 baris

Setelah filter: 112.081 baris

File final ~127MB

5. Dataset Split & Balancing

Split:

Train: 70%

Val: 15%

Test: 15%

Sebelum balancing (train):

non-hoax: 114.987

hoax: 8.367

Setelah balancing:

non-hoax: 114.987

hoax: 114.987

Val dan Test tidak di-balancing.

6. Training Model IndoBERT
6.1 Model & Tokenizer

Model: indolem/indobert-base-uncased

Max length: 256

padding: max_length

truncation: true

6.2 Hyperparameter
batch_size = 64
eval_batch_size = 256
gradient_accumulation = 2
learning_rate = 2e-5
weight_decay = 0.01
epochs = 3
seed = 42
load_best_model_at_end = True


Training berjalan ±2 jam di GPU T4.

6.3 Training Summary

steps: 3594

final training loss ≈ 0.0085

best checkpoint: checkpoint-3594

7. Hasil Evaluasi Model
7.1 Validation Set
Accuracy:         0.9983
Precision hoax:   0.9921
Recall hoax:      0.9833
F1 hoax:          0.9877


Confusion Matrix – Validation

Masukkan gambarnya ke:

preview/confusion-matrix-validation.png

7.2 Test Set
Accuracy:         0.9983
Precision hoax:   0.9938
Recall hoax:      0.9810
F1 hoax:          0.9874


Confusion Matrix – Test

Masukkan gambarnya ke:

preview/confusion-matrix-test.png

7.3 Kesimpulan

Model sangat akurat pada kedua kelas.

Sensitivitas tinggi terhadap pola hoaks.

False positive dan false negative sangat rendah.

Sangat stabil antara validation & test → generalisasi bagus.

8. Risk Level Thresholds

Backend menggunakan probabilitas hoaks untuk membuat kategori:

P(hoax)	Risk	Penjelasan
> 0.98	high	Model sangat yakin ini hoaks
0.60 – 0.98	medium	Mirip pola hoaks, perlu cek ulang
≤ 0.60	low	Cenderung bukan hoaks

Backend juga mengembalikan:

risk_level
risk_explanation

9. Backend API (FastAPI)
9.1 Endpoint /health
{ "status": "ok" }

9.2 Endpoint /predict

Body:

{
  "text": "berita yang ingin dicek"
}


Output:

{
  "label": "hoax",
  "score": 0.991,
  "probabilities": {
    "hoax": 0.991,
    "not_hoax": 0.009
  },
  "risk_level": "high",
  "risk_explanation": "Model sangat yakin ini hoaks..."
}

10. Frontend (Vercel)
10.1 Fitur

Input teks berita

Hasil prediksi + probability

Risk level badge (high/medium/low)

Loading spinner

Tombol Copy hasil

Tombol Share link

Dark mode toggle

Error handling ramah pengguna




11. Struktur Folder Frontend
Deteksi_Hoaks/
│── public/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│── preview/
│   ├── confusion-matrix-validation.png
│   └── confusion-matrix-test.png
│── package.json
│── README.md

12. Pengembangan Lanjutan

Batch prediction (upload CSV)

Deteksi gambar hoaks (OCR + CLIP)

Cross-check otomatis ke TurnBackHoax

Model ensemble untuk stabilitas lebih tinggi

Logging aktivitas pengguna anonim untuk memperbaiki threshold

13. Disclaimer

Model ini:

Bukan pengganti verifikasi fakta profesional

Tidak boleh dipakai sebagai satu-satunya dasar keputusan

Hanya membantu mendeteksi pola hoaks umum

Tetap periksa ke sumber resmi (CekFakta, TurnBackHoax, Kominfo)
