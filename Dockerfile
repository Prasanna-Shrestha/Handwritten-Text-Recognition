FROM python:3.10-slim

# System deps: tesseract, OpenCV runtime, unzip, wget, fonts (optional)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install gdown


# Copy application code
COPY . .

# (Optional) Env lets you override model path if structure inside zip differs
ENV NOTE_BUDDY_MODEL_PATH=/app/model/checkpoint-5080

EXPOSE 8000

# Start FastAPI (prod)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
