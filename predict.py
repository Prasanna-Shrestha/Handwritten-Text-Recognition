from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import pytesseract
import os
import io
from datetime import datetime
import numpy as np
import zipfile
import gdown

# Load model FOR LOCAL HOSTING
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# MODEL_PATH = "model/checkpoint-5080"
# model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "checkpoint-5080")
MODEL_URL = "https://drive.google.com/uc?id=1z9gKcNF7EWzlJaJ0rLImc6M46YN7IIve"
ZIP_PATH = os.path.join(MODEL_DIR, "models.zip")


def download_and_extract_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading from Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)

        gdown.download(MODEL_URL, ZIP_PATH, quiet=False)

        print("Download complete. Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        print("Model extracted successfully.")
    else:
        print("Model already exists. Skipping download.")

download_and_extract_model()

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# MODEL_PATH = "model/model/checkpoint-5080"
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

def output_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"words_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def predict_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_cv = cv2.cvtColor(cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    recent_output_folder = output_folder()

    custom_oem_psm_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(image_cv, config=custom_oem_psm_config, output_type=pytesseract.Output.DICT)

    num_boxes = len(data['text'])
    count = 0

    for i in range(num_boxes):
        word = data['text'][i].strip()
        if word != "":
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            crop = image_cv[y:y+h, x:x+w]
            word_path = os.path.join(recent_output_folder, f'word_{count:04}.png')
            cv2.imwrite(word_path, crop)
            count += 1

    sentence = ""

    for word_image_file in sorted(os.listdir(recent_output_folder)):
        if word_image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')):
            image_path = os.path.join(recent_output_folder, word_image_file)

            try:
                with Image.open(image_path) as img:
                    word_image = img.convert("RGB")

                w, h = word_image.size
                if w < 20 or h < 20:
                    continue

                pixel_values = processor(images=word_image, return_tensors="pt").pixel_values

                if pixel_values.ndim != 4 or pixel_values.shape[2] < 10 or pixel_values.shape[3] < 10:
                    continue

                with torch.no_grad():
                    generated_ids = model.generate(pixel_values)
                    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                sentence += predicted_text.strip() + " "

            except Exception:
                continue

    return sentence.strip()
