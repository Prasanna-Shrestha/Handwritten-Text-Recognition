from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import pytesseract
import io
import numpy as np

app = FastAPI(title="Handwritten OCR API")

# Load model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
MODEL_PATH = "/app/model/checkpoint-5080"
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

# Tiling thresholds
MAX_WIDTH = 1600
MAX_HEIGHT = 1600

def tile_image(image, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """Split large image into smaller tiles if too big."""
    w, h = image.size
    if w <= max_width and h <= max_height:
        return [image]
    
    tiles = []
    for top in range(0, h, max_height):
        for left in range(0, w, max_width):
            box = (left, top, min(left + max_width, w), min(top + max_height, h))
            tiles.append(image.crop(box))
    return tiles

def segment_and_recognize(image_cv):
    """Segment words and recognize text for a given image (OpenCV format)."""
    sentence = ""
    config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(image_cv, config=config, output_type=pytesseract.Output.DICT)

    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word != "":
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            crop = image_cv[y:y+h, x:x+w]
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            try:
                word_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                pixel_values = processor(images=word_image, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values)
                    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                sentence += predicted_text.strip() + " "
            except:
                continue
    return sentence.strip()

def predict_from_image(image_bytes):
    """Main prediction function with tiling for large images."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tiles = tile_image(image)
    full_text = ""

    for tile in tiles:
        image_cv = cv2.cvtColor(np.array(tile), cv2.COLOR_RGB2BGR)
        text = segment_and_recognize(image_cv)
        full_text += text + " "
    
    return full_text.strip()

@app.get("/")
def home():
    return {"message": "Handwritten OCR API is running."}

@app.get("/healthz")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = predict_from_image(contents)
        return JSONResponse(content={"recognized_text": text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
