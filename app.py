from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predict import load_models, predict_from_image

app = FastAPI(title="Handwritten OCR API - Stable")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor, model = load_models()

@app.get("/")
def home():
    return {"message": "API running. Use POST /predict."}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file.")
    try:
        image_bytes = await file.read()
        result = predict_from_image(image_bytes, processor, model)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
