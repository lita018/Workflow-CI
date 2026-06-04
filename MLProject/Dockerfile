FROM python:3.11-slim

WORKDIR /app

# Install dependencies sistem yang diperlukan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements dan install library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mlserver mlserver-mlflow

# Copy seluruh isi project dan model hasil training
COPY . .

# Environment variable untuk MLflow
ENV MLFLOW_DISABLE_ENV_CREATION=true

EXPOSE 8080

# Command untuk menjalankan model serving via MLserver (bawaan MLflow)
CMD ["mlserver", "start", "."]