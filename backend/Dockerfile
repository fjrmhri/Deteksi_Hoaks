FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_ID="fjrmhri/Deteksi_Hoaks" \
    MODEL_SUBFOLDER="" \
    MAX_LENGTH=256 \
    PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]