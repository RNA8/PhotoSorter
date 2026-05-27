import math
import numpy as np
import cv2
from PIL import Image


def score_image_quality(path: str) -> dict:
    img_bgr = _load_bgr(path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    f = np.fft.fft2(gray.astype(np.float32))
    magnitude = np.abs(np.fft.fftshift(f))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    total = magnitude.sum() + 1e-6
    center = magnitude[cy - r:cy + r, cx - r:cx + r].sum()
    blur_score = float((total - center) / total)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    exposure = float(max(0.0, 1.0 - hist[240:].sum() - hist[:16].sum()))

    return {'sharpness_raw': sharpness_raw, 'blur_score': blur_score, 'exposure': exposure}


def normalize_sharpness(scores: list) -> list:
    max_raw = max(s['sharpness_raw'] for s in scores) if scores else 1.0
    for s in scores:
        s['sharpness'] = s['sharpness_raw'] / (max_raw + 1e-6)
    return scores


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is not None:
        return img
    pil = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
