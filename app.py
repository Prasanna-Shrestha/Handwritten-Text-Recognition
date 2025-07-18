import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import load_models, predict_from_image

app = FastAPI(title="NoteBuddy OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load models at startup (once)
processor, model = load_models()


@app.get("/")
def home():
    return {"message": "NoteBuddy OCR API is running. Use POST /predict."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        image_bytes = await file.read()
        result = predict_from_image(image_bytes, processor, model)
        return result 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
