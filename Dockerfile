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

# Download model at build time (ONE TIME)
# NOTE: this downloads to /app/model/
RUN mkdir -p /app/model && \
    gdown --id 1z9gKcNF7EWzlJaJ0rLImc6M46YN7IIve -O /app/model/models.zip && \
    unzip /app/model/models.zip -d /app/model && \
    rm /app/model/models.zip

# Copy application code
COPY . .

# (Optional) Env lets you override model path if structure inside zip differs
ENV NOTE_BUDDY_MODEL_PATH=/app/model/checkpoint-5080

EXPOSE 8000

# Start FastAPI (prod)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
