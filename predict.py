import os
import io
import time
from typing import Tuple, Dict, Any, List

import torch
import numpy as np
import cv2
import pytesseract
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_PATH = os.getenv("NOTE_BUDDY_MODEL_PATH", "/app/model/checkpoint-5080")
BASE_PROCESSOR_ID = os.getenv("NOTE_BUDDY_PROCESSOR_ID", "microsoft/trocr-base-handwritten")

MAX_WIDTH = int(os.getenv("NOTE_BUDDY_MAX_WIDTH", "2000"))  # px


def load_models() -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """
    Loads TrOCR processor + fine-tuned model from disk.
    Called once at startup in app.py.
    """
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model path '{MODEL_PATH}' not found in container. "
            "Confirm Docker build downloaded & unzipped the model."
        )

    processor = TrOCRProcessor.from_pretrained(BASE_PROCESSOR_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    model.eval()
    return processor, model


def load_image_dual(image_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if MAX_WIDTH > 0 and pil_img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / float(pil_img.width)
        new_h = int(pil_img.height * ratio)
        pil_img = pil_img.resize((MAX_WIDTH, new_h), Image.LANCZOS)

    cv_img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img, cv_img_bgr


def get_word_boxes(cv_img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (x1, y1, x2, y2) word boxes.
    """
    config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(
        cv_img_bgr,
        config=config,
        output_type=pytesseract.Output.DICT
    )
    boxes = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        if not txt:
            continue
        try:
            x, y, w, h = (
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["width"][i]),
                int(data["height"][i]),
            )
        except Exception:
            continue
        if w < 5 or h < 5:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def infer_word(pil_word, processor, model) -> str:
    w_, h_ = pil_word.size
    if w_ < 20 or h_ < 20:
        return ""

    pixel_values = processor(images=pil_word, return_tensors="pt").pixel_values
    if pixel_values.ndim != 4 or pixel_values.shape[-1] < 10 or pixel_values.shape[-2] < 10:
        return ""

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return pred.strip()


def predict_from_image(image_bytes: bytes, processor, model) -> Dict[str, Any]:
    tic = time.time()

    pil_img, cv_img_bgr = load_image_dual(image_bytes)

    boxes = get_word_boxes(cv_img_bgr)

    words_out = []
    for (x1, y1, x2, y2) in boxes:
        crop_bgr = cv_img_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        pred = infer_word(crop_pil, processor, model)
        if pred:
            words_out.append(pred)

    sentence = " ".join(words_out).strip()
    elapsed = time.time() - tic

    return {
        "text": sentence,
        "words_detected": len(boxes),
        "words_predicted": len(words_out),
        "latency_s": round(elapsed, 3),
    }
