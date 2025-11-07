# ==========================
# 1) Persiapan Lingkungan
# ==========================
print("# ==========================")
print("# 1) Persiapan Lingkungan")
print("# ==========================")

import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_HEAD_REQUEST", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

try:
    from huggingface_hub import constants as hf_constants
except Exception:
    hf_constants = None

paket_wajib = {
    "torch": "torch==2.2.0",
    "transformers": "transformers==4.38.2",
    "datasets": "datasets==2.17.1",
    "accelerate": "accelerate==0.27.2",
    "pandas": "pandas==2.1.4",
    "numpy": "numpy==1.26.4",
    "scikit-learn": "scikit-learn==1.3.2",
    "tqdm": "tqdm==4.66.2",
    "matplotlib": "matplotlib==3.8.2",
    "imbalanced-learn": "imbalanced-learn==0.11.0",
    "openpyxl": "openpyxl==3.1.2",
    "sentencepiece": "sentencepiece==0.1.99",
    "shap": "shap==0.45.1",
    "fastapi": "fastapi==0.110.2",
    "uvicorn": "uvicorn[standard]==0.29.0",
    "pydantic": "pydantic==2.7.1"
}

paket_instal = {k: v for k, v in paket_wajib.items() if k != "torch"}
print("Memastikan versi paket yang dibutuhkan terpasang.")

pip_base = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    "--upgrade",
]

gpu_dir = os.path.exists("/proc/driver/nvidia/version")

torch_perintah_gpu = pip_base + [
    paket_wajib["torch"],
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu118",
]
torch_perintah_cpu = pip_base + [
    paket_wajib["torch"],
    "--index-url",
    "https://download.pytorch.org/whl/cpu",
]

if gpu_dir:
    try:
        subprocess.check_call(torch_perintah_gpu)
    except subprocess.CalledProcessError:
        print("Instalasi torch CUDA gagal, mencoba versi CPU.")
        subprocess.check_call(torch_perintah_cpu)
else:
    print("GPU tidak terdeteksi pada sistem, memasang torch versi CPU.")
    subprocess.check_call(torch_perintah_cpu)

if paket_instal:
    pip_command = pip_base + list(paket_instal.values())
    subprocess.check_call(pip_command)

if hf_constants is not None:
    hf_constants.HF_HUB_DISABLE_HEAD_REQUEST = True
    hf_constants.HF_HUB_ENABLE_HF_TRANSFER = True

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset
from fastapi import FastAPI
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    perangkat = torch.device("cuda")
    nama_gpu = torch.cuda.get_device_name(perangkat)
    mem_total = torch.cuda.get_device_properties(perangkat).total_memory / (1024 ** 3)
    print(f"GPU aktif: {nama_gpu} ({mem_total:.2f} GB)")
    bebas, total = torch.cuda.mem_get_info()
    print(f"Perkiraan memori bebas: {bebas / (1024 ** 3):.2f} GB dari {total / (1024 ** 3):.2f} GB")
else:
    perangkat = torch.device("cpu")
    print("GPU tidak terdeteksi, menggunakan CPU.")

# ==========================
# 2) Konfigurasi
# ==========================
print("\n# ==========================")
print("# 2) Konfigurasi")
print("# ==========================")

cfg = {
    "jalur_data": "dataset/berita_hoaks.csv",
    "nama_model": "indolem/indobert-base-uncased",
    "panjang_maks": 256,
    "ukuran_batch_latih": 16,
    "ukuran_batch_eval": 64,
    "akumulasi_gradien": 2,
    "laju_pembelajaran": 2e-5,
    "epoh": 4,
    "proporsi_validasi": 0.15,
    "proporsi_uji": 0.15,
    "fp16": True,
    "gradien_klip": 1.0,
    "warmup_proporsi": 0.1,
    "acak_tetap": True,
    "nilai_acak": 42,
    "strategi_imbalance": "bobot",
    "direktori_model": "model_terbaik",
    "metrik_patokan": "f1",
    "patience_henti_awal": 2,
    "num_workers": 2,
    "persistent_workers": True,
    "plot_kurva": True,
    "tampilkan_roc": False,
    "jumlah_contoh_tinjau": 5,
    "maks_batch_latih": None,
    "maks_batch_eval": None,
    "gunakan_cross_val": True,
    "jumlah_folds_domain": 4,
    "epoh_validasi_domain": 1,
    "maks_batch_latih_cv": 4,
    "maks_batch_eval_cv": 8,
    "limit_sample_shap": 4,
    "ukuran_batch_inferensi_demo": 32
}


def adaptasi_cfg(cfg_awal: Dict[str, object]) -> Dict[str, object]:
    cfg_baru = dict(cfg_awal)
    cfg_baru["fp16"] = cfg_baru.get("fp16", False) and perangkat.type == "cuda"
    if perangkat.type != "cuda":
        if cfg_baru.get("maks_batch_latih") is None:
            cfg_baru["maks_batch_latih"] = 2
            print("Menjalankan pada CPU: batasi batch latih per epoh untuk efisiensi.")
        if cfg_baru.get("maks_batch_eval") is None:
            cfg_baru["maks_batch_eval"] = 4
            print("Menjalankan pada CPU: batasi batch evaluasi per epoh.")
        if cfg_baru.get("num_workers", 0) != 0:
            cfg_baru["num_workers"] = 0
            print("Menonaktifkan worker tambahan DataLoader pada CPU.")
        if cfg_baru.get("persistent_workers", False):
            cfg_baru["persistent_workers"] = False
        if cfg_baru.get("epoh", 1) > 1:
            cfg_baru["epoh"] = 1
            print("Menetapkan epoh=1 untuk demonstrasi cepat di CPU.")
        if cfg_baru.get("nama_model") == "indolem/indobert-base-uncased":
            cfg_baru["nama_model"] = "cahya/distilbert-base-indonesian"
            print("Menggunakan model DistilBERT Indonesia untuk eksekusi CPU.")
    return cfg_baru


cfg = adaptasi_cfg(cfg)

if cfg["acak_tetap"]:
    print("Mengatur seed acak seragam.")
    random.seed(cfg["nilai_acak"])
    np.random.seed(cfg["nilai_acak"])
    torch.manual_seed(cfg["nilai_acak"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["nilai_acak"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.benchmark = True

print("Ringkasan konfigurasi:")
for k, v in cfg.items():
    print(f"  {k}: {v}")

# ==========================
# 3) Muat dan Validasi Data
# ==========================
print("\n# ==========================")
print("# 3) Muat dan Validasi Data")
print("# ==========================")

jalur_data = Path(cfg["jalur_data"])
jalur_data.parent.mkdir(parents=True, exist_ok=True)

if not jalur_data.exists():
    print(f"Berkas {jalur_data} tidak ditemukan. Menyiapkan dari sumber Excel.")
    sumber_excel = Path("dataset") / "Cleaned"
    berkas_excel = sorted(sumber_excel.glob("*.xlsx"))
    if not berkas_excel:
        raise FileNotFoundError(
            "Tidak menemukan dataset Excel di dataset/Cleaned. Pastikan repositori berisi data yang diperlukan."
        )

    def pilih_kolom_teks(df: pd.DataFrame) -> str:
        kandidat = ["Clean Narasi", "Narasi", "isi_berita", "teks", "text", "isi", "artikel", "judul"]
        for nama in kandidat:
            if nama in df.columns:
                return nama
        return df.columns[0]

    himpunan = []
    for path in berkas_excel:
        df_excel = pd.read_excel(path)
        kol_teks = pilih_kolom_teks(df_excel)
        if "hoax" not in df_excel.columns:
            raise ValueError(f"File {path.name} tidak memiliki kolom 'hoax'.")
        subset = df_excel[[kol_teks, "hoax"]].copy()
        subset.columns = ["teks", "label"]
        subset["teks"] = subset["teks"].astype(str)
        subset["label"] = subset["label"].astype(int)
        subset["sumber"] = path.stem
        himpunan.append(subset)

    gabungan = pd.concat(himpunan, ignore_index=True)
    gabungan = gabungan.dropna(subset=["teks"])
    gabungan = gabungan[gabungan["teks"].str.strip().str.len() > 0]
    gabungan = gabungan.drop_duplicates(subset=["teks", "label"])
    gabungan.to_csv(jalur_data, index=False)
    print(f"Dataset gabungan disimpan ke {jalur_data} dengan {len(gabungan)} baris.")

print("Memuat dataset utama...")
data = pd.read_csv(jalur_data)
kolom_teks = "teks"
kolom_label = "label"

if kolom_teks not in data.columns or kolom_label not in data.columns:
    raise ValueError("Dataset wajib memiliki kolom 'teks' dan 'label'.")

if "sumber" not in data.columns:
    kandidat = [kol for kol in data.columns if "sumber" in kol.lower() or "source" in kol.lower()]
    if kandidat:
        data["sumber"] = data[kandidat[0]].astype(str)
    else:
        data["sumber"] = "tidak_diketahui"
        print("Peringatan: kolom sumber tidak ditemukan, seluruh baris diberi label 'tidak_diketahui'.")
else:
    data["sumber"] = data["sumber"].astype(str)

print(f"Jumlah baris awal: {len(data)}")
print("Membersihkan duplikat dan nilai kosong...")
data = data.drop_duplicates(subset=[kolom_teks]).dropna(subset=[kolom_teks, kolom_label])

if set(data[kolom_label].unique()) - {0, 1}:
    raise ValueError("Label harus bernilai 0 (bukan hoaks) atau 1 (hoaks).")

print(f"Jumlah baris setelah pembersihan dasar: {len(data)}")

# ==========================
# 4) Pembersihan Kebocoran Label dan Augmentasi
# ==========================
print("\n# ==========================")
print("# 4) Pembersihan Kebocoran Label dan Augmentasi")
print("# ==========================")

import re

pola_kebocoran = {
    "tag_salah": re.compile(r"\[salah\]", re.IGNORECASE),
    "tag_hoaks": re.compile(r"\[hoaks\]", re.IGNORECASE),
    "cek_fakta": re.compile(r"\bcek fakta\b", re.IGNORECASE),
    "cek_fakta_en": re.compile(r"\bfact check\b", re.IGNORECASE),
    "hoaks_kata": re.compile(r"\bhoaks\b", re.IGNORECASE),
    "hoax_kata": re.compile(r"\bhoax\b", re.IGNORECASE)
}
pola_url = re.compile(r"https?://\S+")
pola_spasi = re.compile(r"\s+")

statistik_pembersihan = Counter()


def bersihkan_teks(teks: str, statistik: Optional[Counter] = None) -> str:
    if not isinstance(teks, str):
        return ""
    teks_baru = teks
    if statistik is not None:
        for nama, pola in pola_kebocoran.items():
            if pola.search(teks_baru):
                statistik[nama] += 1
            teks_baru = pola.sub(" ", teks_baru)
    else:
        for pola in pola_kebocoran.values():
            teks_baru = pola.sub(" ", teks_baru)
    teks_baru = pola_url.sub(" ", teks_baru)
    teks_baru = teks_baru.replace("\n", " ")
    teks_baru = teks_baru.lower()
    teks_baru = pola_spasi.sub(" ", teks_baru).strip()
    return teks_baru


print("Membersihkan teks dan menghapus kata kunci pemicu kebocoran label...")
data[kolom_teks] = data[kolom_teks].astype(str)
data[kolom_teks] = data[kolom_teks].apply(lambda t: bersihkan_teks(t, statistik_pembersihan))
print("Ringkasan kata kunci yang dibersihkan:")
for nama, jumlah in statistik_pembersihan.items():
    print(f"  {nama}: {jumlah}")

print("Menghapus entri kosong pasca pembersihan...")
data = data[data[kolom_teks].str.len() > 0]
print(f"Jumlah baris setelah pembersihan lanjutan: {len(data)}")

print("Menambahkan variasi informal dan memastikan keseimbangan kelas...")

contoh_informal = [
    {"teks": "Bro seriusan vaksin bikin magnet? udah pada tau belom?", "label": 1, "sumber": "sintetik_sosmed"},
    {"teks": "Katanya bulan depan BBM gratis, padahal berita lama dibikin ulang", "label": 1, "sumber": "sintetik_sosmed"},
    {"teks": "Video lama banjir disebar ulang bilang ibukota tenggelam minggu ini", "label": 1, "sumber": "sintetik_sosmed"},
    {"teks": "Thread panjang soal chip di vaksin, jangan langsung percaya dong", "label": 1, "sumber": "sintetik_forum"},
    {"teks": "Temen gue bilang PLN matiin listrik seminggu, ternyata klarifikasi bilang hoaks", "label": 1, "sumber": "sintetik_forum"},
    {"teks": "Selamat pagi, jadwal vaksin booster gratis ada di Puskesmas kecamatan ya", "label": 0, "sumber": "sintetik_sosmed"},
    {"teks": "Pemkab bagikan paket sembako resmi, cek situsnya buat daftar", "label": 0, "sumber": "sintetik_sosmed"},
    {"teks": "Diskon listrik diperpanjang sesuai siaran pers PLN, bukan kabar burung", "label": 0, "sumber": "sintetik_forum"},
    {"teks": "BNPB rilis update gempa, info lengkap ada di akun resmi", "label": 0, "sumber": "sintetik_forum"},
    {"teks": "Jangan panik, polisi klarifikasi isu penculikan anak yang sempat viral", "label": 0, "sumber": "sintetik_rumor"}
]

augment_df = pd.DataFrame(contoh_informal)
augment_df[kolom_teks] = augment_df[kolom_teks].apply(lambda t: bersihkan_teks(t, statistik_pembersihan))
data = pd.concat([data, augment_df], ignore_index=True)

if cfg["strategi_imbalance"] == "oversampling":
    ros = RandomOverSampler(random_state=cfg["nilai_acak"])
    teks_res, label_res = ros.fit_resample(data[[kolom_teks, "sumber"]], data[kolom_label])
    data = pd.DataFrame({kolom_teks: teks_res[kolom_teks], kolom_label: label_res, "sumber": teks_res["sumber"]})
    print("Oversampling dilakukan untuk menyeimbangkan kelas.")

print(f"Total baris setelah augmentasi: {len(data)}")

distribusi_label = data[kolom_label].value_counts().sort_index()
print("Distribusi label:")
for label_nilai, jumlah in distribusi_label.items():
    nama = "bukan hoaks" if label_nilai == 0 else "hoaks"
    persentase = jumlah / len(data) * 100
    print(f"  {label_nilai} ({nama}): {jumlah} ({persentase:.2f}%)")

print("Distribusi domain (sumber) dan label:")
domain_label = data.groupby("sumber")[kolom_label].value_counts().unstack(fill_value=0)
print(domain_label.sort_index())

# ==========================
# 5) Persiapan Tokenizer dan Utilitas Dataset
# ==========================
print("\n# ==========================")
print("# 5) Persiapan Tokenizer dan Utilitas Dataset")
print("# ==========================")

from functools import lru_cache


@lru_cache(maxsize=1)
def unduh_model_hf(nama_model: str, cache_root: Path = Path("hf_cache")) -> Path:
    kandidat_lokal = Path(nama_model)
    if kandidat_lokal.exists() and kandidat_lokal.is_dir():
        return kandidat_lokal

    tujuan = cache_root / nama_model.replace("/", "_")
    tujuan.mkdir(parents=True, exist_ok=True)

    file_wajib = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt"
    ]
    file_opsional = {"tokenizer.json", "added_tokens.json"}
    dasar_url = f"https://huggingface.co/{nama_model}/resolve/main/"

    for nama_berkas in file_wajib + sorted(file_opsional):
        target = tujuan / nama_berkas
        if target.exists():
            continue
        url = dasar_url + nama_berkas
        print(f"Mengunduh {nama_berkas} dari Hugging Face...")
        percobaan = 0
        while True:
            try:
                resp = requests.get(
                    url,
                    stream=True,
                    timeout=120,
                    headers={"User-Agent": "hoax-detector/1.0"}
                )
            except requests.RequestException as exc:
                if percobaan < 2:
                    jeda = 2 ** percobaan
                    print(f"  Koneksi gagal ({exc}), mencoba lagi dalam {jeda} dtk.")
                    time.sleep(jeda)
                    percobaan += 1
                    continue
                raise RuntimeError(f"Gagal terhubung ke Hugging Face untuk {nama_berkas}.") from exc

            if resp.status_code == 404 and nama_berkas in file_opsional:
                print(f"  {nama_berkas} tidak ditemukan, dilewati.")
                break

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                if resp.status_code >= 500 and percobaan < 2:
                    jeda = 2 ** percobaan
                    print(f"  Server {resp.status_code} ({exc}); mencoba lagi dalam {jeda} dtk.")
                    time.sleep(jeda)
                    percobaan += 1
                    continue
                raise RuntimeError(
                    f"Gagal mengunduh {nama_berkas} (status {resp.status_code})."
                ) from exc

            with open(target, "wb") as f:
                for potongan in resp.iter_content(chunk_size=8192):
                    if potongan:
                        f.write(potongan)
            break

    return tujuan


lokasi_model = unduh_model_hf(cfg["nama_model"])
print(f"Model akan dimuat dari: {lokasi_model}")

tokenizer_global = AutoTokenizer.from_pretrained(lokasi_model)
kolator_global = DataCollatorWithPadding(tokenizer=tokenizer_global)


def buat_dataset(dataframe: pd.DataFrame, tokenizer: AutoTokenizer, cfg_pakai: Dict[str, object]) -> Dataset:
    dataset = Dataset.from_pandas(dataframe.reset_index(drop=True))
    dataset = dataset.map(
        lambda contoh: tokenizer(
            contoh[kolom_teks],
            truncation=True,
            max_length=cfg_pakai["panjang_maks"],
            padding=False
        ),
        batched=True,
        remove_columns=[kolom_teks],
        desc="Tokenisasi"
    )
    if kolom_label in dataset.column_names:
        dataset = dataset.rename_column(kolom_label, "labels")
    kolom_sisa = [kol for kol in dataset.column_names if kol.startswith("__index_")]
    if kolom_sisa:
        dataset = dataset.remove_columns(kolom_sisa)
    dataset.set_format(type="torch")
    return dataset


def buat_dataloader(dataset: Dataset, ukuran_batch: int, acak: bool, cfg_pakai: Dict[str, object]) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=ukuran_batch,
        shuffle=acak,
        collate_fn=kolator_global,
        num_workers=cfg_pakai["num_workers"],
        pin_memory=perangkat.type == "cuda",
        persistent_workers=cfg_pakai["persistent_workers"] if cfg_pakai["num_workers"] > 0 else False
    )


label2id = {"bukan_hoaks": 0, "hoaks": 1}
id2label = {0: "bukan_hoaks", 1: "hoaks"}


class MemoriTidakCukupError(RuntimeError):
    pass



def evaluasi(model_eval, loader, cfg_pakai: Dict[str, object]):
    model_eval.eval()
    semua_label, semua_pred, semua_prob = [], [], []
    total_loss = 0.0
    total_data = 0
    maks_eval = cfg_pakai.get("maks_batch_eval")
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            if maks_eval is not None and batch_id >= maks_eval:
                print(f"Batas {maks_eval} batch evaluasi tercapai.")
                break
            label_batch = batch.pop("labels").to(perangkat)
            batch = {k: v.to(perangkat) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=cfg_pakai["fp16"] and perangkat.type == "cuda"):
                keluaran = model_eval(**batch)
                loss = cfg_pakai["bobot_kerugian"](keluaran.logits, label_batch)
            total_loss += loss.item() * label_batch.size(0)
            total_data += label_batch.size(0)
            pred = torch.argmax(keluaran.logits, dim=1)
            prob = torch.softmax(keluaran.logits, dim=-1)
            semua_label.extend(label_batch.cpu().tolist())
            semua_pred.extend(pred.cpu().tolist())
            semua_prob.extend(prob.cpu().tolist())
    total_data = max(total_data, 1)
    rata_loss = total_loss / total_data
    akurasi = accuracy_score(semua_label, semua_pred)
    presisi = precision_score(semua_label, semua_pred, zero_division=0)
    recall = recall_score(semua_label, semua_pred, zero_division=0)
    f1 = f1_score(semua_label, semua_pred, zero_division=0)
    return rata_loss, akurasi, presisi, recall, f1, semua_label, semua_pred, semua_prob


def latih_model(train_df: pd.DataFrame,
                valid_df: pd.DataFrame,
                cfg_latih: Dict[str, object],
                nama_sesi: str,
                test_df: Optional[pd.DataFrame] = None,
                simpan_model: bool = False) -> Dict[str, object]:
    print(f"\nMulai sesi pelatihan: {nama_sesi}")
    cfg_latih = dict(cfg_latih)
    cfg_latih["fp16"] = cfg_latih.get("fp16", False)

    dataset_latih = buat_dataset(train_df[[kolom_teks, kolom_label]], tokenizer_global, cfg_latih)
    dataset_valid = buat_dataset(valid_df[[kolom_teks, kolom_label]], tokenizer_global, cfg_latih)
    loader_latih = buat_dataloader(dataset_latih, cfg_latih["ukuran_batch_latih"], True, cfg_latih)
    loader_valid = buat_dataloader(dataset_valid, cfg_latih["ukuran_batch_eval"], False, cfg_latih)
    loader_test = None
    if test_df is not None:
        dataset_test = buat_dataset(test_df[[kolom_teks, kolom_label]], tokenizer_global, cfg_latih)
        loader_test = buat_dataloader(dataset_test, cfg_latih["ukuran_batch_eval"], False, cfg_latih)

    bobot_kelas = None
    if cfg_latih.get("strategi_imbalance") == "bobot":
        nilai_bobot = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_df[kolom_label]
        )
        bobot_kelas = torch.tensor(nilai_bobot, dtype=torch.float32, device=perangkat)
        print(f"Menggunakan bobot kelas: {nilai_bobot}")
    else:
        print("Tidak menggunakan bobot kelas khusus.")

    model = AutoModelForSequenceClassification.from_pretrained(
        lokasi_model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    model.to(perangkat)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_latih["laju_pembelajaran"], weight_decay=0.01)
    langkah_total = math.ceil(len(loader_latih) / cfg_latih["akumulasi_gradien"]) * cfg_latih["epoh"]
    langkah_warmup = int(langkah_total * cfg_latih["warmup_proporsi"])
    penjadwal = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=langkah_warmup,
        num_training_steps=langkah_total
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg_latih["fp16"] and perangkat.type == "cuda")
    cfg_latih["bobot_kerugian"] = nn.CrossEntropyLoss(weight=bobot_kelas)

    riwayat = {"epoh": [], "latih_loss": [], "valid_loss": [], "valid_f1": []}
    terbaik = {"nilai": -float("inf"), "epoh": -1, "state_dict": None, "loss": float("inf")}
    kesabaran = cfg_latih["patience_henti_awal"]

    percobaan_selesai = False
    while not percobaan_selesai:
        try:
            for epoh in range(cfg_latih["epoh"]):
                model.train()
                total_loss = 0.0
                langkah = 0
                pbar = tqdm(loader_latih, desc=f"Epoh {epoh + 1}/{cfg_latih['epoh']} - latih {nama_sesi}", leave=False)
                optimizer.zero_grad(set_to_none=True)
                for batch_id, batch in enumerate(pbar):
                    if cfg_latih.get("maks_batch_latih") is not None and batch_id >= cfg_latih["maks_batch_latih"]:
                        print(f"Batas {cfg_latih['maks_batch_latih']} batch latih tercapai pada epoh {epoh + 1}.")
                        break
                    label_batch = batch.pop("labels").to(perangkat)
                    batch = {k: v.to(perangkat) for k, v in batch.items()}
                    try:
                        with torch.cuda.amp.autocast(enabled=cfg_latih["fp16"] and perangkat.type == "cuda"):
                            keluaran = model(**batch)
                            loss = cfg_latih["bobot_kerugian"](keluaran.logits, label_batch)
                        loss = loss / cfg_latih["akumulasi_gradien"]
                        scaler.scale(loss).backward()
                    except RuntimeError as err:
                        if "out of memory" in str(err).lower():
                            torch.cuda.empty_cache()
                            raise MemoriTidakCukupError from err
                        raise

                    if (batch_id + 1) % cfg_latih["akumulasi_gradien"] == 0 or (batch_id + 1) == len(loader_latih):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_latih["gradien_klip"])
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        penjadwal.step()

                    total_loss += loss.item() * label_batch.size(0) * cfg_latih["akumulasi_gradien"]
                    langkah += label_batch.size(0)
                    pbar.set_postfix({"loss": total_loss / max(langkah, 1)})

                rata_loss_latih = total_loss / max(len(loader_latih.dataset), 1)
                val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluasi(model, loader_valid, cfg_latih)
                riwayat["epoh"].append(epoh + 1)
                riwayat["latih_loss"].append(rata_loss_latih)
                riwayat["valid_loss"].append(val_loss)
                riwayat["valid_f1"].append(val_f1)
                print(
                    f"Epoh {epoh + 1} [{nama_sesi}]: loss_latih={rata_loss_latih:.4f} | loss_valid={val_loss:.4f} | "
                    f"akurasi={val_acc:.4f} | presisi={val_prec:.4f} | recall={val_rec:.4f} | f1={val_f1:.4f}"
                )

                metrik = val_f1 if cfg_latih["metrik_patokan"] == "f1" else val_acc
                if metrik > terbaik["nilai"]:
                    terbaik["nilai"] = metrik
                    terbaik["epoh"] = epoh + 1
                    terbaik["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
                    terbaik["loss"] = val_loss
                    kesabaran = cfg_latih["patience_henti_awal"]
                    print(f"Model terbaik sementara diperbarui pada epoh {epoh + 1}.")
                else:
                    kesabaran -= 1
                    print(f"Henti awal: kesabaran tersisa {kesabaran} epoh.")
                    if kesabaran == 0:
                        print("Henti awal diaktifkan.")
                        percobaan_selesai = True
                        break
            percobaan_selesai = True
        except MemoriTidakCukupError:
            if perangkat.type != "cuda":
                raise
            if cfg_latih["ukuran_batch_latih"] == 1 and cfg_latih["akumulasi_gradien"] >= 8:
                raise RuntimeError("Tidak dapat mengurangi batch lebih lanjut setelah OOM.")
            print("Peringatan: CUDA OOM terdeteksi. Menyesuaikan parameter pelatihan.")
            cfg_latih["ukuran_batch_latih"] = max(1, cfg_latih["ukuran_batch_latih"] // 2)
            cfg_latih["akumulasi_gradien"] = min(cfg_latih["akumulasi_gradien"] * 2, 8)
            loader_latih = buat_dataloader(dataset_latih, cfg_latih["ukuran_batch_latih"], True, cfg_latih)
            loader_valid = buat_dataloader(dataset_valid, cfg_latih["ukuran_batch_eval"], False, cfg_latih)
            if loader_test is not None:
                loader_test = buat_dataloader(dataset_test, cfg_latih["ukuran_batch_eval"], False, cfg_latih)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_latih["laju_pembelajaran"], weight_decay=0.01)
            langkah_total = math.ceil(len(loader_latih) / cfg_latih["akumulasi_gradien"]) * cfg_latih["epoh"]
            langkah_warmup = int(langkah_total * cfg_latih["warmup_proporsi"])
            penjadwal = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=langkah_warmup,
                num_training_steps=langkah_total
            )
            scaler = torch.cuda.amp.GradScaler(enabled=cfg_latih["fp16"] and perangkat.type == "cuda")
            print(
                f"Ukuran batch latih baru: {cfg_latih['ukuran_batch_latih']} (efektif {cfg_latih['ukuran_batch_latih'] * cfg_latih['akumulasi_gradien']})."
            )

    if terbaik["state_dict"] is not None:
        model.load_state_dict({k: v.to(perangkat) for k, v in terbaik["state_dict"].items()})
        print(f"Bobot terbaik dari epoh {terbaik['epoh']} telah dimuat untuk evaluasi {nama_sesi}.")
    else:
        print("Tidak ada bobot terbaik yang tersimpan.")

    hasil = {
        "riwayat": riwayat,
        "terbaik": terbaik,
        "model": model if simpan_model else None,
        "tokenizer": tokenizer_global if simpan_model else None,
        "valid_metrics": None,
        "test_metrics": None,
        "loader_valid": loader_valid,
        "loader_test": loader_test,
        "loss_fn": cfg_latih["bobot_kerugian"]
    }

    val_loss, val_acc, val_prec, val_rec, val_f1, val_label, val_pred, val_prob = evaluasi(model, loader_valid, cfg_latih)
    hasil["valid_metrics"] = {
        "loss": val_loss,
        "accuracy": val_acc,
        "precision": val_prec,
        "recall": val_rec,
        "f1": val_f1,
        "labels": val_label,
        "predictions": val_pred,
        "probabilities": val_prob
    }
    print(
        f"Validasi {nama_sesi}: loss={val_loss:.4f} | akurasi={val_acc:.4f} | presisi={val_prec:.4f} | "
        f"recall={val_rec:.4f} | f1={val_f1:.4f}"
    )

    if loader_test is not None:
        uji_loss, uji_acc, uji_prec, uji_rec, uji_f1, test_label, test_pred, test_prob = evaluasi(model, loader_test, cfg_latih)
        hasil["test_metrics"] = {
            "loss": uji_loss,
            "accuracy": uji_acc,
            "precision": uji_prec,
            "recall": uji_rec,
            "f1": uji_f1,
            "labels": test_label,
            "predictions": test_pred,
            "probabilities": test_prob
        }
        print(
            f"Uji {nama_sesi}: loss={uji_loss:.4f} | akurasi={uji_acc:.4f} | presisi={uji_prec:.4f} | "
            f"recall={uji_rec:.4f} | f1={uji_f1:.4f}"
        )

    if not simpan_model:
        model.to("cpu")
        torch.cuda.empty_cache()

    return hasil


# ==========================
# 6) Validasi Silang Berbasis Domain
# ==========================
print("\n# ==========================")
print("# 6) Validasi Silang Berbasis Domain")
print("# ==========================")

cv_hasil = []
if cfg.get("gunakan_cross_val", False):
    jumlah_domain = data["sumber"].nunique()
    n_splits = min(cfg["jumlah_folds_domain"], jumlah_domain)
    if n_splits < 2:
        print("Jumlah domain unik kurang dari dua, validasi silang domain dilewati.")
    else:
        gkf = GroupKFold(n_splits=n_splits)
        cfg_cv = dict(cfg)
        cfg_cv["epoh"] = min(cfg["epoh_validasi_domain"], cfg["epoh"])
        if cfg.get("maks_batch_latih_cv") is not None:
            cfg_cv["maks_batch_latih"] = cfg["maks_batch_latih_cv"]
        if cfg.get("maks_batch_eval_cv") is not None:
            cfg_cv["maks_batch_eval"] = cfg["maks_batch_eval_cv"]
        print(f"Melakukan GroupKFold dengan {n_splits} lipatan berdasarkan domain.")
        for fold_ke, (train_idx, valid_idx) in enumerate(gkf.split(data, data[kolom_label], groups=data["sumber"]), start=1):
            train_fold = data.iloc[train_idx].reset_index(drop=True)
            valid_fold = data.iloc[valid_idx].reset_index(drop=True)
            domain_valid = valid_fold["sumber"].unique().tolist()
            print(f"Fold {fold_ke}: validasi pada domain {domain_valid}")
            hasil_fold = latih_model(train_fold, valid_fold, cfg_cv, f"Fold-{fold_ke}")
            cv_hasil.append({
                "fold": fold_ke,
                "domain_valid": domain_valid,
                "metrics": hasil_fold["valid_metrics"]
            })
else:
    print("Validasi silang domain dinonaktifkan melalui konfigurasi.")

if cv_hasil:
    rata_f1 = np.mean([h["metrics"]["f1"] for h in cv_hasil])
    rata_acc = np.mean([h["metrics"]["accuracy"] for h in cv_hasil])
    print("Ringkasan validasi silang domain:")
    for entri in cv_hasil:
        metrik = entri["metrics"]
        print(
            f"  Fold {entri['fold']} (domain {entri['domain_valid']}): "
            f"akurasi={metrik['accuracy']:.4f}, presisi={metrik['precision']:.4f}, "
            f"recall={metrik['recall']:.4f}, f1={metrik['f1']:.4f}"
        )
    print(f"  Rata-rata akurasi: {rata_acc:.4f}")
    print(f"  Rata-rata F1: {rata_f1:.4f}")

# ==========================
# 7) Pembagian Data Hold-out Berbasis Domain
# ==========================
print("\n# ==========================")
print("# 7) Pembagian Data Hold-out Berbasis Domain")
print("# ==========================")

gss = GroupShuffleSplit(n_splits=1, test_size=cfg["proporsi_uji"], random_state=cfg["nilai_acak"])
train_idx, test_idx = next(gss.split(data, data[kolom_label], groups=data["sumber"]))
train_temp = data.iloc[train_idx].reset_index(drop=True)
test_df = data.iloc[test_idx].reset_index(drop=True)
print(f"Data latih sementara: {len(train_temp)} baris")
print(f"Data uji domain hold-out: {len(test_df)} baris")

val_splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["proporsi_validasi"] / (1 - cfg["proporsi_uji"]), random_state=cfg["nilai_acak"]) 
train_idx2, valid_idx2 = next(val_splitter.split(train_temp, train_temp[kolom_label], groups=train_temp["sumber"]))
train_df = train_temp.iloc[train_idx2].reset_index(drop=True)
valid_df = train_temp.iloc[valid_idx2].reset_index(drop=True)
print(f"Data latih akhir: {len(train_df)} baris")
print(f"Data validasi domain: {len(valid_df)} baris")

# ==========================
# 8) Pelatihan Final dan Evaluasi Hold-out
# ==========================
print("\n# ==========================")
print("# 8) Pelatihan Final dan Evaluasi Hold-out")
print("# ==========================")

hasil_final = latih_model(train_df, valid_df, cfg, "Pelatihan-Utama", test_df=test_df, simpan_model=True)
model = hasil_final["model"]
riwayat = hasil_final["riwayat"]
terbaik = hasil_final["terbaik"]

uji_metrics = hasil_final["test_metrics"]
if uji_metrics is None:
    raise RuntimeError("Evaluasi uji tidak tersedia meskipun test_df diberikan.")

print("Membuat laporan klasifikasi untuk data uji hold-out...")
laporan = classification_report(uji_metrics["labels"], uji_metrics["predictions"], output_dict=True, zero_division=0)
laporan_df = pd.DataFrame(laporan).T.rename(index={
    "0": "bukan hoaks",
    "1": "hoaks",
    "accuracy": "akurasi",
    "macro avg": "rata-rata makro",
    "weighted avg": "rata-rata berbobot"
})
if "support" in laporan_df.columns:
    laporan_df.rename(columns={"precision": "presisi", "recall": "recall", "f1-score": "f1", "support": "jumlah"}, inplace=True)
    laporan_df["jumlah"] = laporan_df["jumlah"].fillna(0).round().astype(int)
print(laporan_df.to_string(float_format=lambda x: f"{x:.4f}"))

matriks = confusion_matrix(uji_metrics["labels"], uji_metrics["predictions"])
plt.figure(figsize=(4, 4))
plt.imshow(matriks, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriks Kebingungan")
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ["Prediksi bukan hoaks", "Prediksi hoaks"])
plt.yticks(ticks, ["Aktual bukan hoaks", "Aktual hoaks"])
for i in range(matriks.shape[0]):
    for j in range(matriks.shape[1]):
        warna = "white" if matriks[i, j] > matriks.max() / 2 else "black"
        plt.text(j, i, format(matriks[i, j], "d"), horizontalalignment="center", color=warna)
plt.ylabel("Label Aktual")
plt.xlabel("Label Prediksi")
plt.tight_layout()
plt.savefig("matriks_kebingungan.png", dpi=200)
plt.close()
print("Matriks kebingungan disimpan ke matriks_kebingungan.png.")

if cfg.get("plot_kurva") and riwayat["epoh"]:
    plt.figure(figsize=(6, 4))
    plt.plot(riwayat["epoh"], riwayat["latih_loss"], label="Loss latih")
    plt.plot(riwayat["epoh"], riwayat["valid_loss"], label="Loss validasi")
    plt.xlabel("Epoh")
    plt.ylabel("Loss")
    plt.title("Kurva Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kurva_loss.png", dpi=200)
    plt.close()
    print("Kurva loss disimpan ke kurva_loss.png.")

if cfg.get("plot_kurva") and riwayat["epoh"]:
    plt.figure(figsize=(6, 4))
    plt.plot(riwayat["epoh"], riwayat["valid_f1"], marker="o", label="F1 validasi")
    plt.xlabel("Epoh")
    plt.ylabel("F1")
    plt.title("Kurva F1 Validasi")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kurva_f1.png", dpi=200)
    plt.close()
    print("Kurva F1 disimpan ke kurva_f1.png.")

# ==========================
# 9) Evaluasi pada Data Eksternal 2025
# ==========================
print("\n# ==========================")
print("# 9) Evaluasi pada Data Eksternal 2025")
print("# ==========================")

data_eksternal = pd.DataFrame([
    {"teks": "April 2025: unggahan viral menyebut presiden melegalkan judi online nasional. Pemerintah membantah kabar itu.", "label": 1, "sumber": "eksternal_apr2025"},
    {"teks": "April 2025: pesan berantai klaim bank akan menghapus saldo tabungan warga saat audit sistem baru.", "label": 1, "sumber": "eksternal_apr2025"},
    {"teks": "Mei 2025: artikel resmi Kemenkes umumkan jadwal vaksinasi booster anak sekolah.", "label": 0, "sumber": "eksternal_mei2025"},
    {"teks": "Mei 2025: BMKG menerbitkan peringatan dini hujan lebat besok dan imbau warga waspada.", "label": 0, "sumber": "eksternal_mei2025"},
    {"teks": "April 2025: akun palsu menulis bahwa semua tol gratis seumur hidup mulai bulan depan.", "label": 1, "sumber": "eksternal_apr2025"},
    {"teks": "Mei 2025: rilis resmi BPS memaparkan pertumbuhan ekonomi triwulan pertama.", "label": 0, "sumber": "eksternal_mei2025"}
])
data_eksternal[kolom_teks] = data_eksternal[kolom_teks].apply(lambda t: bersihkan_teks(t, statistik_pembersihan))

print("Evaluasi terhadap data eksternal yang tidak terlihat saat pelatihan...")

cfg_eval = dict(cfg)
cfg_eval["maks_batch_eval"] = None
cfg_eval["bobot_kerugian"] = hasil_final["loss_fn"] if hasil_final.get("loss_fn") is not None else nn.CrossEntropyLoss()

dataset_ext = buat_dataset(data_eksternal[[kolom_teks, kolom_label]], tokenizer_global, cfg_eval)
loader_ext = buat_dataloader(dataset_ext, cfg["ukuran_batch_eval"], False, cfg_eval)
ext_loss, ext_acc, ext_prec, ext_rec, ext_f1, ext_labels, ext_pred, _ = evaluasi(model, loader_ext, cfg_eval)
print(
    f"Eksternal 2025: loss={ext_loss:.4f} | akurasi={ext_acc:.4f} | presisi={ext_prec:.4f} | "
    f"recall={ext_rec:.4f} | f1={ext_f1:.4f}"
)

# ==========================
# 10) Analisis Interpretabilitas
# ==========================
print("\n# ==========================")
print("# 10) Analisis Interpretabilitas")
print("# ==========================")

import shap

masker = shap.maskers.Text(tokenizer_global)
label_index_hoaks = label2id["hoaks"]


def fungsi_prediksi(teks_list: Iterable[str]) -> np.ndarray:
    masukan = tokenizer_global(
        list(teks_list),
        truncation=True,
        max_length=cfg["panjang_maks"],
        padding=True,
        return_tensors="pt"
    ).to(perangkat)
    with torch.no_grad():
        logits = model(**masukan).logits
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
    return prob


contoh_shap = test_df.sample(n=min(cfg["limit_sample_shap"], len(test_df)), random_state=cfg["nilai_acak"])
print(f"Menghitung SHAP untuk {len(contoh_shap)} sampel uji.")
explainer = shap.Explainer(fungsi_prediksi, masker)
shap_values = explainer(contoh_shap[kolom_teks].tolist())

for teks_asli, shap_info in zip(contoh_shap[kolom_teks], shap_values):
    token_data = shap_info.data
    nilai_hoaks = shap_info.values[:, label_index_hoaks]
    pasangan = list(zip(token_data, nilai_hoaks))
    pasangan = [p for p in pasangan if p[0].strip()]
    pasangan.sort(key=lambda x: abs(x[1]), reverse=True)
    print("-" * 60)
    print(f"Teks: {teks_asli[:120]}...")
    print("Kontribusi token teratas terhadap kelas 'hoaks':")
    for token, nilai in pasangan[:10]:
        arah = "mendukung" if nilai > 0 else "mengurangi"
        print(f"  {token}: {nilai:.4f} ({arah})")

print("Menganalisis atensi Transformer pada contoh representatif...")
contoh_atensi = test_df.iloc[0][kolom_teks]
masukan_atensi = tokenizer_global(contoh_atensi, return_tensors="pt", truncation=True, max_length=cfg["panjang_maks"]).to(perangkat)
with torch.no_grad():
    keluaran_atensi = model(**masukan_atensi, output_attentions=True)
attentions = keluaran_atensi.attentions  # tuple(layer) dengan bentuk (batch, head, seq, seq)
att_tensor = torch.stack(attentions).mean(dim=(0, 1)).squeeze(0)
cls_attention = att_tensor[0]
cls_attention = cls_attention / cls_attention.sum()
token_ids = masukan_atensi["input_ids"][0].cpu().tolist()
token_kata = tokenizer_global.convert_ids_to_tokens(token_ids)
plt.figure(figsize=(10, 4))
plt.bar(range(len(token_kata)), cls_attention.cpu().numpy())
plt.xticks(range(len(token_kata)), token_kata, rotation=90)
plt.ylabel("Bobot atensi terhadap [CLS]")
plt.title("Distribusi atensi pada contoh uji")
plt.tight_layout()
plt.savefig("atensi_cls.png", dpi=200)
plt.close()
print("Visualisasi atensi disimpan ke atensi_cls.png.")

# ==========================
# 11) Optimalisasi Inferensi dan API
# ==========================
print("\n# ==========================")
print("# 11) Optimalisasi Inferensi dan API")
print("# ==========================")


def prediksi_batch(daftar_teks: List[str]) -> List[Dict[str, object]]:
    masukan = tokenizer_global(
        daftar_teks,
        truncation=True,
        max_length=cfg["panjang_maks"],
        padding=True,
        return_tensors="pt"
    ).to(perangkat)
    with torch.no_grad():
        logits = model(**masukan).logits
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
    hasil = []
    for teks, nilai in zip(daftar_teks, prob):
        indeks = int(np.argmax(nilai))
        label_pred = id2label[indeks]
        status = "hoaks" if label_pred == "hoaks" else "bukan hoaks"
        hasil.append({
            "teks": teks,
            "label": status,
            "skor": float(nilai[indeks])
        })
    return hasil


def ukur_kinerja_inferensi(jumlah: int = cfg["ukuran_batch_inferensi_demo"]):
    sampel = list(test_df[kolom_teks].sample(n=min(jumlah, len(test_df)), random_state=cfg["nilai_acak"]))
    mulai = time.time()
    _ = prediksi_batch(sampel)
    durasi = time.time() - mulai
    rata = durasi / max(len(sampel), 1)
    print(f"Inferensi batch berisi {len(sampel)} teks selesai dalam {durasi:.2f} dtk (rata-rata {rata:.4f} dtk/teks).")


ukur_kinerja_inferensi()


class PermintaanPrediksi(BaseModel):
    teks: List[str]


class ResponPrediksi(BaseModel):
    hasil: List[Dict[str, object]]


def buat_aplikasi_fastapi(model_ref, tokenizer_ref):
    app = FastAPI(title="API Deteksi Hoaks", version="1.0")

    @app.post("/prediksi", response_model=ResponPrediksi)
    def prediksi_endpoint(permintaan: PermintaanPrediksi):
        return ResponPrediksi(hasil=prediksi_batch(permintaan.teks))

    @app.get("/status")
    def status():
        return {"status": "siap", "model": cfg["nama_model"]}

    return app


app_fastapi = buat_aplikasi_fastapi(model, tokenizer_global)
print("Contoh API FastAPI siap. Jalankan uvicorn secara terpisah untuk mengaktifkan layanan.")

# ==========================
# 12) Penyimpanan Model dan Fungsi Inferensi Tunggal
# ==========================
print("\n# ==========================")
print("# 12) Penyimpanan Model dan Fungsi Inferensi Tunggal")
print("# ==========================")


def simpan_model_dan_artefak():
    direktori_model = Path(cfg["direktori_model"])
    direktori_model.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(direktori_model)
    tokenizer_global.save_pretrained(direktori_model)
    info_label = {"id2label": id2label, "label2id": label2id}
    with open(direktori_model / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(info_label, f, ensure_ascii=False, indent=2)
    pd.Series(cfg).to_json(direktori_model / "konfigurasi.json")
    pd.DataFrame(riwayat).to_csv(direktori_model / "riwayat_pelatihan.csv", index=False)
    if cv_hasil:
        pd.DataFrame([
            {
                "fold": entri["fold"],
                "domain_valid": ", ".join(entri["domain_valid"]),
                "akurasi": entri["metrics"]["accuracy"],
                "presisi": entri["metrics"]["precision"],
                "recall": entri["metrics"]["recall"],
                "f1": entri["metrics"]["f1"]
            }
            for entri in cv_hasil
        ]).to_csv(direktori_model / "validasi_silang_domain.csv", index=False)
    print(f"Model dan artefak disimpan ke {direktori_model.resolve()}.")


simpan_model_dan_artefak()


def prediksi_hoaks(teks: str) -> Dict[str, object]:
    hasil = prediksi_batch([teks])[0]
    return hasil


contoh_teks_baru = [
    "Pesan berantai menyebutkan pemerintah akan mematikan listrik nasional selama tiga hari untuk menangkap penjahat.",
    "Pemerintah daerah mengumumkan jadwal vaksinasi gratis di puskesmas setempat."
]

print("Prediksi pada contoh teks baru:")
for hasil in prediksi_batch(contoh_teks_baru):
    print("-" * 60)
    print(f"Teks: {hasil['teks']}")
    print(f"Prediksi: {hasil['label']} (skor {hasil['skor']:.4f})")

print("Ringkasan akhir:")
print(
    f"  Metrik uji hold-out - Akurasi: {uji_metrics['accuracy']:.4f}, Presisi: {uji_metrics['precision']:.4f}, "
    f"Recall: {uji_metrics['recall']:.4f}, F1: {uji_metrics['f1']:.4f}"
)
print(
    f"  Data eksternal 2025 - Akurasi: {ext_acc:.4f}, Presisi: {ext_prec:.4f}, Recall: {ext_rec:.4f}, F1: {ext_f1:.4f}"
)
print(f"  Bobot terbaik berasal dari epoh {terbaik['epoh']} dengan {cfg['metrik_patokan']} {terbaik['nilai']:.4f}.")
print(f"  Model tersimpan di: {Path(cfg['direktori_model']).resolve()}")
print("  Gunakan fungsi prediksi_hoaks(teks) atau prediksi_batch([...]) untuk inferensi cepat.")

print("Masukkan teks berita yang ingin diuji. Ketik 'exit' untuk berhenti.")

while True:
    try:
        teks_uji = input("Teks berita: ")
        if teks_uji.lower() == 'exit':
            print("Keluar dari mode prediksi.")
            break
        if not teks_uji.strip():
            print("Input kosong. Mohon masukkan teks berita.")
            continue

        hasil = prediksi_hoaks(teks_uji)
        print(f"Hasil prediksi: {hasil['label']} dengan keyakinan {hasil['skor']:.4f}")
        print("-" * 60)

    except EOFError:
        print("\nInterupsi terdeteksi. Keluar dari mode prediksi.")
        break
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        print("Mohon coba lagi.")
