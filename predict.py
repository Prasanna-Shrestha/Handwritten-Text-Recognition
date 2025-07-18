import os
import io
import time
from typing import Tuple, Dict, Any, List

import numpy as np
import cv2
import pytesseract
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_PATH = os.getenv("NOTE_BUDDY_MODEL_PATH", "/app/model/checkpoint-5080")
BASE_PROCESSOR_ID = os.getenv("NOTE_BUDDY_PROCESSOR_ID", "microsoft/trocr-base-handwritten")

MAX_IMAGE_DIM = int(os.getenv("NOTE_BUDDY_MAX_IMAGE_DIM", "1600"))  # px;
MIN_WORD_W = int(os.getenv("NOTE_BUDDY_MIN_WORD_W", "20"))
MIN_WORD_H = int(os.getenv("NOTE_BUDDY_MIN_WORD_H", "20"))
MAX_WORDS  = int(os.getenv("NOTE_BUDDY_MAX_WORDS", "200"))


def load_models():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model path '{MODEL_PATH}' not found. Check Docker unzip and NOTE_BUDDY_MODEL_PATH."
        )
    processor = TrOCRProcessor.from_pretrained(BASE_PROCESSOR_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    model.eval()
    return processor, model


def _load_image(image_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    cv_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img, cv_bgr


def _word_boxes(cv_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(cv_bgr, config=config, output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data["text"])):
        txt = data["text"][i]
        if not txt or not txt.strip():
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if w < MIN_WORD_W or h < MIN_WORD_H:
            continue
        boxes.append((x, y, x + w, y + h))
        if len(boxes) >= MAX_WORDS:
            break
    return boxes


def _infer_batch(crops: List[Image.Image], processor, model) -> List[str]:
    if not crops:
        return []
    pixel_values = processor(images=crops, return_tensors="pt", padding=True).pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [p.strip() for p in preds]


def predict_from_image(image_bytes: bytes, processor, model) -> Dict[str, Any]:
    start = time.time()

    _, cv_bgr = _load_image(image_bytes)
    boxes = _word_boxes(cv_bgr)

    crops = []
    for (x1, y1, x2, y2) in boxes:
        crop_bgr = cv_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crops.append(Image.fromarray(crop_rgb))

    preds = _infer_batch(crops, processor, model)
    sentence = " ".join([p for p in preds if p])

    return {
        "text": sentence,
        "words_detected": len(boxes),
        "words_predicted": len(preds),
        "latency_s": round(time.time() - start, 3),
    }
