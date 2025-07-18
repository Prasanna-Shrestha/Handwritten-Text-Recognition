import os
import io
import tempfile
from datetime import datetime

import torch
import cv2
import numpy as np
import pytesseract
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Model Load (global, once at startup)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
MODEL_PATH = "/app/model/checkpoint-5080"
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
model.eval()  # inference mode


# ------------------------------------------------------------------
# Utility: convert bytes -> both PIL + OpenCV image
# ------------------------------------------------------------------
def load_image_dual(image_bytes: bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # OpenCV wants BGR uint8 array
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img, cv_img


# ------------------------------------------------------------------

# ------------------------------------------------------------------
def predict_from_image(image_bytes: bytes) -> str:
    pil_img, cv_img_bgr = load_image_dual(image_bytes)

    custom_oem_psm_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(
        cv_img_bgr,
        config=custom_oem_psm_config,
        output_type=pytesseract.Output.DICT
    )

    n = len(data["text"])
    words_out = []
    with tempfile.TemporaryDirectory(prefix="words_", dir="/tmp") as temp_dir:
        for i in range(n):
            raw_txt = data["text"][i].strip()
            if not raw_txt:
                continue
            try:
                x, y, w, h = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
            except Exception:
                continue


            if w < 20 or h < 20:
                continue

            crop_bgr = cv_img_bgr[y : y + h, x : x + w]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            crop_path = os.path.join(temp_dir, f"word_{i:04}.png")
            crop_pil.save(crop_path)

            cw, ch = crop_pil.size
            if cw < 20 or ch < 20:
                continue

            pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values
            if pixel_values.ndim != 4 or pixel_values.shape[-1] < 10 or pixel_values.shape[-2] < 10:
                continue

            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            predicted_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            if predicted_text.strip():
                words_out.append(predicted_text.strip())

    return " ".join(words_out).strip()