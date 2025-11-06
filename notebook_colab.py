# ==========================
# 1) Persiapan Lingkungan
# ==========================
print("# ==========================")
print("# 1) Persiapan Lingkungan")
print("# ==========================")

import subprocess
import sys

paket_wajib = {
    "torch": "torch==2.1.2",  # mendukung CUDA 11.8 di Colab T4
    "transformers": "transformers==4.38.2",
    "datasets": "datasets==2.17.1",
    "accelerate": "accelerate==0.27.2",
    "pandas": "pandas==2.1.4",
    "numpy": "numpy==1.26.4",
    "scikit-learn": "scikit-learn==1.3.2",
    "tqdm": "tqdm==4.66.2",
    "matplotlib": "matplotlib==3.8.2",
    "imbalanced-learn": "imbalanced-learn==0.11.0"
}

paket_instal = list(paket_wajib.values())
print("Memastikan versi paket yang dibutuhkan terpasang.")

# ``sys.executable -m pip`` works both in standard Python environments and in
# Google Colab.  A previous revision mistakenly tried to call ``-m !pip`` which
# causes ``ModuleNotFoundError`` because ``!`` is shell syntax that should not be
# used when invoking modules via ``python -m``.  Building the command list
# explicitly avoids that issue and keeps the invocation compatible across
# platforms.
pip_command = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    "--upgrade",
    *paket_instal,
]
subprocess.check_call(pip_command)

import torch

if torch.cuda.is_available():
    perangkat = torch.device("cuda")
    nama_gpu = torch.cuda.get_device_name(perangkat)
    mem_total = torch.cuda.get_device_properties(perangkat).total_memory / (1024 ** 3)
    print(f"GPU aktif: {nama_gpu} ({mem_total:.2f} GB)")
    bebas, total = torch.cuda.mem_get_info()
    print(f"Perkiraan memori bebas: {bebas / (1024 ** 3):.2f} GB dari {total / (1024 ** 3):.2f} GB")
    print("Gunakan !nvidia-smi di Colab untuk detail tambahan.")
else:
    perangkat = torch.device("cpu")
    print("GPU tidak terdeteksi, menggunakan CPU.")

# ==========================
# 2) Konfigurasi
# ==========================
print("\n# ==========================")
print("# 2) Konfigurasi")
print("# ==========================")

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
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
    "jumlah_contoh_tinjau": 5
}

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
    print(f"Berkas {jalur_data} tidak ditemukan. Membuat contoh sintetis.")
    contoh_data = pd.DataFrame(
        {
            "teks": [
                "Vaksin COVID-19 mengandung chip pelacak yang akan mengendalikan pikiran manusia.",
                "Kementerian Kesehatan memastikan vaksin COVID-19 aman dan tidak mengandung chip.",
                "Video tentang bank tutup permanen adalah rekayasa lama yang kembali disebarkan.",
                "Bank Indonesia menyatakan layanan perbankan berjalan normal.",
                "Hoaks telur plastik kembali muncul di media sosial.",
                "Produsen makanan menegaskan telur yang beredar asli dan aman dikonsumsi."
            ],
            "label": [1, 0, 1, 0, 1, 0]
        }
    )
    contoh_data.to_csv(jalur_data, index=False)
    print("Contoh dataset sintetis selesai dibuat.")

data = pd.read_csv(jalur_data)
kolom_teks = "teks"
kolom_label = "label"

if kolom_teks not in data.columns or kolom_label not in data.columns:
    raise ValueError("Dataset wajib memiliki kolom 'teks' dan 'label'.")

print(f"Jumlah baris awal: {len(data)}")
print("Membersihkan duplikat dan nilai kosong...")
data = data.drop_duplicates(subset=[kolom_teks]).dropna(subset=[kolom_teks, kolom_label])

if set(data[kolom_label].unique()) - {0, 1}:
    raise ValueError("Label harus bernilai 0 (bukan hoaks) atau 1 (hoaks).")

print(f"Jumlah baris setelah pembersihan: {len(data)}")

distribusi_label = data[kolom_label].value_counts().sort_index()
print("Distribusi label:")
for label_nilai, jumlah in distribusi_label.items():
    nama = "bukan hoaks" if label_nilai == 0 else "hoaks"
    persentase = jumlah / len(data) * 100
    print(f"  {label_nilai} ({nama}): {jumlah} ({persentase:.2f}%)")

print("Membagi data menjadi latih/validasi/uji secara stratified...")
train_df, temp_df = train_test_split(
    data,
    test_size=cfg["proporsi_validasi"] + cfg["proporsi_uji"],
    random_state=cfg["nilai_acak"],
    stratify=data[kolom_label]
)
proporsi_uji_koreksi = cfg["proporsi_uji"] / (cfg["proporsi_validasi"] + cfg["proporsi_uji"])
valid_df, test_df = train_test_split(
    temp_df,
    test_size=proporsi_uji_koreksi,
    random_state=cfg["nilai_acak"],
    stratify=temp_df[kolom_label]
)

print(f"Data latih: {len(train_df)} baris")
print(f"Data validasi: {len(valid_df)} baris")
print(f"Data uji: {len(test_df)} baris")

# ==========================
# 4) Pra-pemrosesan dan Tokenisasi
# ==========================
print("\n# ==========================")
print("# 4) Pra-pemrosesan dan Tokenisasi")
print("# ==========================")


def bersihkan_teks(teks: str) -> str:
    if not isinstance(teks, str):
        return ""
    teks = teks.lower().strip()
    return " ".join(teks.split())


train_df = train_df.assign(teks=train_df[kolom_teks].apply(bersihkan_teks))
valid_df = valid_df.assign(teks=valid_df[kolom_teks].apply(bersihkan_teks))
test_df = test_df.assign(teks=test_df[kolom_teks].apply(bersihkan_teks))

if cfg["strategi_imbalance"] == "oversampling":
    print("Melakukan oversampling pada data latih.")
    ros = RandomOverSampler(random_state=cfg["nilai_acak"])
    teks_res, label_res = ros.fit_resample(train_df[[kolom_teks]], train_df[kolom_label])
    train_df = pd.DataFrame({kolom_teks: teks_res[kolom_teks], kolom_label: label_res})
    print(f"Jumlah baris latih setelah oversampling: {len(train_df)}")

print("Memuat tokenizer Indonesia...")
tokenizer = AutoTokenizer.from_pretrained(cfg["nama_model"])


def buat_dataset(dataframe: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(dataframe.reset_index(drop=True))
    dataset = dataset.map(
        lambda contoh: tokenizer(
            contoh[kolom_teks],
            truncation=True,
            max_length=cfg["panjang_maks"],
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


dataset_latih = buat_dataset(train_df[[kolom_teks, kolom_label]])
dataset_valid = buat_dataset(valid_df[[kolom_teks, kolom_label]])
dataset_uji = buat_dataset(test_df[[kolom_teks, kolom_label]])

print("Dataset Hugging Face siap digunakan.")

# ==========================
# 5) Pembuatan DataLoader
# ==========================
print("\n# ==========================")
print("# 5) Pembuatan DataLoader")
print("# ==========================")

kolator = DataCollatorWithPadding(tokenizer=tokenizer)


def buat_dataloader(dataset: Dataset, ukuran_batch: int, acak: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=ukuran_batch,
        shuffle=acak,
        collate_fn=kolator,
        num_workers=cfg["num_workers"],
        pin_memory=perangkat.type == "cuda",
        persistent_workers=cfg["persistent_workers"] if cfg["num_workers"] > 0 else False
    )


loader_latih = buat_dataloader(dataset_latih, cfg["ukuran_batch_latih"], True)
loader_valid = buat_dataloader(dataset_valid, cfg["ukuran_batch_eval"], False)
loader_uji = buat_dataloader(dataset_uji, cfg["ukuran_batch_eval"], False)

ukuran_batch_efektif = cfg["ukuran_batch_latih"] * cfg["akumulasi_gradien"]
print(f"Ukuran batch latih: {cfg['ukuran_batch_latih']} (efektif {ukuran_batch_efektif})")
print(f"Ukuran batch evaluasi: {cfg['ukuran_batch_eval']}")

# ==========================
# 6) Inisialisasi Model
# ==========================
print("\n# ==========================")
print("# 6) Inisialisasi Model")
print("# ==========================")

label2id = {"bukan_hoaks": 0, "hoaks": 1}
id2label = {0: "bukan_hoaks", 1: "hoaks"}

model = AutoModelForSequenceClassification.from_pretrained(
    cfg["nama_model"],
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
model.to(perangkat)

bobot_kelas = None
if cfg["strategi_imbalance"] == "bobot":
    nilai_bobot = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df[kolom_label]
    )
    bobot_kelas = torch.tensor(nilai_bobot, dtype=torch.float32, device=perangkat)
    print(f"Menggunakan bobot kelas: {nilai_bobot}")
else:
    print("Tidak menggunakan bobot kelas khusus.")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["laju_pembelajaran"], weight_decay=0.01)
langkah_total = math.ceil(len(loader_latih) / cfg["akumulasi_gradien"]) * cfg["epoh"]
langkah_warmup = int(langkah_total * cfg["warmup_proporsi"])
penjadwal = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=langkah_warmup,
    num_training_steps=langkah_total
)
scaler = torch.cuda.amp.GradScaler(enabled=cfg["fp16"] and perangkat.type == "cuda")
bobot_kerugian = nn.CrossEntropyLoss(weight=bobot_kelas)

print("Model siap dilatih.")

# ==========================
# 7) Pelatihan dengan Pemantauan
# ==========================
print("\n# ==========================")
print("# 7) Pelatihan dengan Pemantauan")
print("# ==========================")


class MemoriTidakCukupError(RuntimeError):
    """Kesalahan khusus untuk menangani OOM."""


def evaluasi(model_eval, loader):
    model_eval.eval()
    semua_label, semua_pred, semua_prob = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            label_batch = batch.pop("labels").to(perangkat)
            batch = {k: v.to(perangkat) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=cfg["fp16"] and perangkat.type == "cuda"):
                keluaran = model_eval(**batch)
                loss = bobot_kerugian(keluaran.logits, label_batch)
            total_loss += loss.item() * label_batch.size(0)
            pred = torch.argmax(keluaran.logits, dim=1)
            prob = torch.softmax(keluaran.logits, dim=-1)
            semua_label.extend(label_batch.cpu().tolist())
            semua_pred.extend(pred.cpu().tolist())
            semua_prob.extend(prob.cpu().tolist())
    rata_loss = total_loss / len(loader.dataset)
    akurasi = accuracy_score(semua_label, semua_pred)
    presisi = precision_score(semua_label, semua_pred, zero_division=0)
    recall = recall_score(semua_label, semua_pred, zero_division=0)
    f1 = f1_score(semua_label, semua_pred, zero_division=0)
    return rata_loss, akurasi, presisi, recall, f1, semua_label, semua_pred, semua_prob


riwayat = {"epoh": [], "latih_loss": [], "valid_loss": [], "valid_f1": []}
terbaik = {"nilai": -float("inf"), "epoh": -1, "state_dict": None, "loss": float("inf")}
kesabaran = cfg["patience_henti_awal"]

percobaan_selesai = False
while not percobaan_selesai:
    try:
        for epoh in range(cfg["epoh"]):
            model.train()
            total_loss = 0.0
            langkah = 0
            pbar = tqdm(loader_latih, desc=f"Epoh {epoh + 1}/{cfg['epoh']} - latih", leave=False)
            optimizer.zero_grad(set_to_none=True)
            for batch_id, batch in enumerate(pbar):
                label_batch = batch.pop("labels").to(perangkat)
                batch = {k: v.to(perangkat) for k, v in batch.items()}
                try:
                    with torch.cuda.amp.autocast(enabled=cfg["fp16"] and perangkat.type == "cuda"):
                        keluaran = model(**batch)
                        loss = bobot_kerugian(keluaran.logits, label_batch)
                    loss = loss / cfg["akumulasi_gradien"]
                    scaler.scale(loss).backward()
                except RuntimeError as err:
                    if "out of memory" in str(err).lower():
                        torch.cuda.empty_cache()
                        raise MemoriTidakCukupError from err
                    raise

                if (batch_id + 1) % cfg["akumulasi_gradien"] == 0 or (batch_id + 1) == len(loader_latih):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradien_klip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    penjadwal.step()

                total_loss += loss.item() * label_batch.size(0) * cfg["akumulasi_gradien"]
                langkah += label_batch.size(0)
                pbar.set_postfix({"loss": total_loss / langkah})

            rata_loss_latih = total_loss / len(loader_latih.dataset)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluasi(model, loader_valid)
            riwayat["epoh"].append(epoh + 1)
            riwayat["latih_loss"].append(rata_loss_latih)
            riwayat["valid_loss"].append(val_loss)
            riwayat["valid_f1"].append(val_f1)
            print(
                f"Epoh {epoh + 1}: loss_latih={rata_loss_latih:.4f} | loss_valid={val_loss:.4f} | "
                f"akurasi={val_acc:.4f} | presisi={val_prec:.4f} | recall={val_rec:.4f} | f1={val_f1:.4f}"
            )

            metrik = val_f1 if cfg["metrik_patokan"] == "f1" else val_acc
            if metrik > terbaik["nilai"]:
                terbaik["nilai"] = metrik
                terbaik["epoh"] = epoh + 1
                terbaik["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
                terbaik["loss"] = val_loss
                kesabaran = cfg["patience_henti_awal"]
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
        if cfg["ukuran_batch_latih"] == 1 and cfg["akumulasi_gradien"] >= 8:
            raise RuntimeError("Tidak dapat mengurangi batch lebih lanjut setelah OOM.")
        print("Peringatan: CUDA OOM terdeteksi. Menyesuaikan parameter pelatihan.")
        cfg["ukuran_batch_latih"] = max(1, cfg["ukuran_batch_latih"] // 2)
        cfg["akumulasi_gradien"] = min(cfg["akumulasi_gradien"] * 2, 8)
        loader_latih = buat_dataloader(dataset_latih, cfg["ukuran_batch_latih"], True)
        loader_valid = buat_dataloader(dataset_valid, cfg["ukuran_batch_eval"], False)
        loader_uji = buat_dataloader(dataset_uji, cfg["ukuran_batch_eval"], False)
        ukuran_batch_efektif = cfg["ukuran_batch_latih"] * cfg["akumulasi_gradien"]
        print(f"Ukuran batch latih baru: {cfg['ukuran_batch_latih']} (efektif {ukuran_batch_efektif}).")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["laju_pembelajaran"], weight_decay=0.01)
        langkah_total = math.ceil(len(loader_latih) / cfg["akumulasi_gradien"]) * cfg["epoh"]
        langkah_warmup = int(langkah_total * cfg["warmup_proporsi"])
        penjadwal = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=langkah_warmup,
            num_training_steps=langkah_total
        )
        scaler = torch.cuda.amp.GradScaler(enabled=cfg["fp16"] and perangkat.type == "cuda")

if terbaik["state_dict"] is not None:
    model.load_state_dict({k: v.to(perangkat) for k, v in terbaik["state_dict"].items()})
    print(f"Bobot terbaik dari epoh {terbaik['epoh']} telah dimuat untuk evaluasi.")
else:
    print("Tidak ada bobot terbaik yang tersimpan.")

# ==========================
# 8) Evaluasi dan Pelaporan
# ==========================
print("\n# ==========================")
print("# 8) Evaluasi dan Pelaporan")
print("# ==========================")

uji_loss, uji_acc, uji_prec, uji_rec, uji_f1, label_uji, pred_uji, prob_uji = evaluasi(model, loader_uji)
print(
    f"Hasil uji: loss={uji_loss:.4f} | akurasi={uji_acc:.4f} | presisi={uji_prec:.4f} | "
    f"recall={uji_rec:.4f} | f1={uji_f1:.4f}"
)

nilai_roc = None
if cfg["tampilkan_roc"]:
    try:
        prob_kelas_positif = [p[1] for p in prob_uji]
        nilai_roc = roc_auc_score(label_uji, prob_kelas_positif)
        print(f"ROC-AUC makro: {nilai_roc:.4f}")
    except Exception as err:
        print(f"ROC-AUC tidak dapat dihitung: {err}")

laporan = classification_report(label_uji, pred_uji, output_dict=True, zero_division=0)
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
print("Laporan klasifikasi:")
print(laporan_df.to_string(float_format=lambda x: f"{x:.4f}"))

matriks = confusion_matrix(label_uji, pred_uji)
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
print("Matriks kebingungan disimpan ke matriks_kebingungan.png.")
plt.close()

if cfg["plot_kurva"] and riwayat["epoh"]:
    plt.figure(figsize=(6, 4))
    plt.plot(riwayat["epoh"], riwayat["latih_loss"], label="Loss latih")
    plt.plot(riwayat["epoh"], riwayat["valid_loss"], label="Loss validasi")
    plt.xlabel("Epoh")
    plt.ylabel("Loss")
    plt.title("Kurva Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kurva_loss.png", dpi=200)
    print("Kurva loss disimpan ke kurva_loss.png.")
    plt.close()

if cfg["plot_kurva"] and riwayat["epoh"]:
    plt.figure(figsize=(6, 4))
    plt.plot(riwayat["epoh"], riwayat["valid_f1"], marker="o", label="F1 validasi")
    plt.xlabel("Epoh")
    plt.ylabel("F1")
    plt.title("Kurva F1 Validasi")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kurva_f1.png", dpi=200)
    print("Kurva F1 disimpan ke kurva_f1.png.")
    plt.close()

print("Contoh prediksi benar/salah:")
contoh_df = test_df.assign(prediksi=pred_uji)
contoh_df["status"] = np.where(contoh_df["label"] == contoh_df["prediksi"], "benar", "salah")
contoh_tampil = contoh_df.sample(n=min(cfg["jumlah_contoh_tinjau"], len(contoh_df)), random_state=cfg["nilai_acak"])
print(contoh_tampil[["teks", "label", "prediksi", "status"]])

# ==========================
# 9) Inferensi dan Penyimpanan Model
# ==========================
print("\n# ==========================")
print("# 9) Inferensi dan Penyimpanan Model")
print("# ==========================")

direktori_model = Path(cfg["direktori_model"])
direktori_model.mkdir(parents=True, exist_ok=True)
model.save_pretrained(direktori_model)
tokenizer.save_pretrained(direktori_model)
info_label = {"id2label": id2label, "label2id": label2id}
with open(direktori_model / "label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(info_label, f, ensure_ascii=False, indent=2)

pd.Series(cfg).to_json(direktori_model / "konfigurasi.json")
pd.DataFrame(riwayat).to_csv(direktori_model / "riwayat_pelatihan.csv", index=False)
print(f"Model dan artefak disimpan ke {direktori_model.resolve()}.")


def prediksi_hoaks(teks: str) -> dict:
    model.eval()
    masukan = tokenizer(
        teks,
        truncation=True,
        max_length=cfg["panjang_maks"],
        padding=True,
        return_tensors="pt"
    ).to(perangkat)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=cfg["fp16"] and perangkat.type == "cuda"):
            keluaran = model(**masukan)
            probabilitas = torch.softmax(keluaran.logits, dim=-1).cpu().numpy()[0]
    indeks = int(np.argmax(probabilitas))
    nama_label = id2label[indeks]
    skor = float(probabilitas[indeks])
    return {"label": nama_label, "skor": skor}


contoh_teks_baru = [
    "Pesan berantai menyebutkan pemerintah akan mematikan listrik nasional selama tiga hari untuk menangkap penjahat.",
    "Pemerintah daerah mengumumkan jadwal vaksinasi gratis di puskesmas setempat."
]

print("Prediksi pada contoh teks baru:")
masukan_batch = tokenizer(
    contoh_teks_baru,
    truncation=True,
    max_length=cfg["panjang_maks"],
    padding=True,
    return_tensors="pt"
).to(perangkat)
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=cfg["fp16"] and perangkat.type == "cuda"):
        keluaran_batch = model(**masukan_batch)
        probabilitas_batch = torch.softmax(keluaran_batch.logits, dim=-1).cpu().numpy()

for teks, prob in zip(contoh_teks_baru, probabilitas_batch):
    indeks = int(np.argmax(prob))
    nama_label = id2label[indeks]
    skor = float(prob[indeks])
    status = "hoaks" if nama_label == "hoaks" else "bukan hoaks"
    print("-" * 60)
    print(f"Teks: {teks}")
    print(f"Prediksi: {status} (skor {skor:.4f})")

print("Ringkasan akhir:")
print(
    f"  Metrik uji terbaik - Akurasi: {uji_acc:.4f}, Presisi: {uji_prec:.4f}, "
    f"Recall: {uji_rec:.4f}, F1: {uji_f1:.4f}"
)
if nilai_roc is not None:
    print(f"  ROC-AUC makro: {nilai_roc:.4f}")
print(f"  Bobot terbaik berasal dari epoh {terbaik['epoh']} dengan {cfg['metrik_patokan']} {terbaik['nilai']:.4f}.")
print(f"  Model tersimpan di: {direktori_model.resolve()}")
print("  Gunakan fungsi prediksi_hoaks(teks) untuk inferensi cepat.")

# ==========================
# 10) Masukan Pengguna untuk Prediksi
# ==========================
print("\n# ==========================")
print("# 10) Masukan Pengguna untuk Prediksi")
print("# ==========================")

teks_uji = input("Masukkan teks berita yang ingin diuji: ")
hasil = prediksi_hoaks(teks_uji)
status_hasil = "hoaks" if hasil["label"] == "hoaks" else "bukan hoaks"
print(f"Hasil prediksi: {status_hasil} dengan keyakinan {hasil['skor']:.4f}")
