FROM python:3.11-slim

# Non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Set workdir
WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode
COPY . .

# Env default (bisa dioverride di Settings → Variables Space)
ENV MODEL_ID="fjrmhri/Deteksi_Hoaks" \
    MODEL_SUBFOLDER="" \
    MAX_LENGTH=256 \
    PORT=7860

EXPOSE 7860

# Jalankan FastAPI via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
