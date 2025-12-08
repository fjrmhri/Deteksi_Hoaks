<p align="center">
  <img src="https://img.shields.io/github/stars/fjrmhri/Deteksi_Hoaks?style=for-the-badge&logo=github&color=8b5cf6" alt="Stars"/>
  <img src="https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Transformers-4.57.1-ffcc00?style=for-the-badge&logo=huggingface" alt="Transformers"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
</p>

# Deteksi Hoaks Indonesia – IndoBERT Hoax Detector

Aplikasi Deteksi Hoaks Indonesia adalah sistem end-to-end berbasis IndoBERT yang mendeteksi apakah sebuah teks berita mengandung indikasi hoaks atau bukan.

## Pipeline Singkat
- Model IndoBERT-base yang di-fine-tune dengan dataset besar hoaks + non-hoaks
- Backend FastAPI yang berjalan di Hugging Face Spaces
- Frontend statis (HTML/JS/CSS) yang bisa dideploy ke Vercel
- Skema risk level berdasarkan probabilitas hoaks: low, medium, high

## Arsitektur Sistem
User → Frontend (Vercel) → FastAPI (HuggingFace Spaces) → IndoBERT Model → Prediksi

## Komponen
- Model: indolem/indobert-base-uncased (fine-tuned)
- Endpoint utama: `/predict`
- Output: label, probabilitas, dan risk level

## Dataset & Pipeline Data
### Dataset yang Digunakan
**Non-hoaks:**
- Summarized_CNN.csv
- Summarized_Detik.csv
- Summarized_Kompas.csv
- merged_clean_filtered_2020plus_halfNaT.csv (hasil merging + cleaning besar, filtering tanggal ≥ 2020, dan sebagian NaT dibuang)

**Hoaks:**
- Summarized_TurnBackHoax.csv (berisi narasi hoaks & debunk)

Setelah diselaraskan, skema final dataset:
- url
- judul
- tanggal
- isi_berita
- Narasi
- Clean Narasi
- hoax   (0=non-hoaks, 1=hoaks)
- summary

## Cleaning dan Merging Data
### Cleaning teks (Clean Narasi)
Pipeline:
- lowercase
- hapus HTML tags
- hapus URL
- hapus emoji, simbol, angka tidak penting
- normalize repeated characters ("hebooooh" → "heboh")
- stopwords removal
- stemming ringan (Bahasa Indonesia)
- slang normalization (yg→yang, gk→nggak, dll)
- spell correction ringan
- trim whitespace
- Jika suatu sumber tidak punya kolom tertentu, diisi "".

### Merging
- Semua CSV dibaca
- Kolom dirapikan
- Dedup berdasarkan (url, judul, Clean Narasi)
- Buang baris kosong
- Buang sebagian NaT (agar dataset lebih kecil & bersih)

## Filter Tanggal & Ukuran Dataset
Dataset besar difilter:
- Semua tanggal < 2020 dibuang
- NaT dikurangi 50% secara random

Contoh perubahan ukuran:
- Sebelum filter: 196.928 baris
- Setelah filter: 112.081 baris
- File final ~127MB

## Dataset Split & Balancing
Split:
- Train: 70%
- Val: 15%
- Test: 15%

Sebelum balancing (train):
- non-hoax: 114.987
- hoax: 8.367

Setelah balancing:
- non-hoax: 114.987
- hoax: 114.987

Val dan Test tidak di-balancing.

## Training Model IndoBERT
### Model & Tokenizer
- Model: indolem/indobert-base-uncased
- Max length: 256
- padding: max_length
- truncation: true

### Hyperparameter
- batch_size = 64
- eval_batch_size = 256
- gradient_accumulation = 2
- learning_rate = 2e-5
- weight_decay = 0.01
- epochs = 3
- seed = 42
- load_best_model_at_end = True

Training berjalan ±2 jam di GPU T4.

### Training Summary
- steps: 3594
- final training loss ≈ 0.0085
- best checkpoint: checkpoint-3594

## Hasil Evaluasi Model
### Validation Set
- Accuracy: 0.9983
- Precision hoax: 0.9921
- Recall hoax: 0.9833
- F1 hoax: 0.9877
- Confusion Matrix – Validation: `preview/confusion-matrix-validation.png`

### Test Set
- Accuracy: 0.9983
- Precision hoax: 0.9938
- Recall hoax: 0.9810
- F1 hoax: 0.9874
- Confusion Matrix – Test: `preview/confusion-matrix-test.png`

## Dokumenatsi
Untuk dokumentasi lengkap dan panduan pengembangan, silakan lihat **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Aset
Semua gambar dan ilustrasi berada di folder `preview/` sesuai permintaan proyek.
