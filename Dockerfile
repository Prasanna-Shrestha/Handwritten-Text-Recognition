FROM python:3.10-slim

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install gdown

# Download and unzip model from Google Drive
RUN gdown --id 1z9gKcNF7EWzlJaJ0rLImc6M46YN7IIve -O models.zip \
 && unzip models.zip -d /app/model \
 && rm models.zip

# Copy all application files
COPY . .

# Expose port 8000
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
