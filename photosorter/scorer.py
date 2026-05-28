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


def score_faces(path: str, face_analyzer) -> dict:
    img_bgr = _load_bgr(path)
    faces = face_analyzer.get(img_bgr)
    if not faces:
        return {'gaze_score': 0.5, 'smile_score': 0.5, 'eyes_score': 0.5}

    gazes, smiles, eyes = [], [], []
    for face in faces:
        pose = getattr(face, 'pose', None)
        if pose is not None:
            yaw, pitch = float(pose[0]), float(pose[1])
            gazes.append(max(0.0, 1.0 - math.sqrt(yaw ** 2 + pitch ** 2) / 45.0))
        else:
            gazes.append(0.5)

        lm = getattr(face, 'landmark_3d_68', None)
        if lm is not None:
            eyes.append(_eye_aspect_ratio_score(lm))
            smiles.append(_smile_score(lm))
        else:
            eyes.append(0.5)
            smiles.append(0.5)

    return {
        'gaze_score': float(np.mean(gazes)),
        'smile_score': float(np.mean(smiles)),
        'eyes_score': float(np.mean(eyes)),
    }


def _eye_aspect_ratio_score(lm: np.ndarray) -> float:
    def ear(indices):
        p = lm[indices]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3]) + 1e-6
        return (v1 + v2) / (2.0 * h)

    avg = (ear([36, 37, 38, 39, 40, 41]) + ear([42, 43, 44, 45, 46, 47])) / 2.0
    return 1.0 if avg > 0.2 else 0.0


def _smile_score(lm: np.ndarray) -> float:
    left_corner = lm[48]
    right_corner = lm[54]
    upper_lip = lm[51]
    mouth_width = np.linalg.norm(right_corner - left_corner) + 1e-6
    avg_corner_y = (left_corner[1] + right_corner[1]) / 2.0
    uplift = float(upper_lip[1] - avg_corner_y)
    return float(min(1.0, max(0.0, uplift / (mouth_width * 0.3))))
