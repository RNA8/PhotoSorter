import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN


def time_window_groups(photos: list, window_minutes: int) -> list:
    if not photos:
        return []
    window_seconds = window_minutes * 60
    groups = [[photos[0]]]
    for photo in photos[1:]:
        if photo['timestamp'] - groups[-1][-1]['timestamp'] > window_seconds:
            groups.append([])
        groups[-1].append(photo)
    return groups


def compute_clip_embeddings(paths: list, model, preprocess, device: str) -> np.ndarray:
    images = [preprocess(Image.open(p).convert('RGB')) for p in paths]
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def cluster_by_visual_similarity(embeddings: np.ndarray, eps: float,
                                  min_samples: int) -> np.ndarray:
    if len(embeddings) == 1:
        return np.array([0])
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(embeddings)
    noise_mask = labels == -1
    if noise_mask.any():
        non_noise = np.where(~noise_mask)[0]
        if non_noise.size == 0:
            labels[:] = 0
        else:
            for i in np.where(noise_mask)[0]:
                sims = embeddings[i] @ embeddings[non_noise].T
                labels[i] = labels[non_noise[np.argmax(sims)]]
    return labels


def group_photos(photos: list, window_minutes: int, eps: float,
                 min_samples: int, model, preprocess, device: str) -> list:
    candidate_groups = time_window_groups(photos, window_minutes)
    moments = []
    for group in candidate_groups:
        if len(group) == 1:
            moments.append(group)
            continue
        paths = [p['path'] for p in group]
        embeddings = compute_clip_embeddings(paths, model, preprocess, device)
        labels = cluster_by_visual_similarity(embeddings, eps, min_samples)
        clusters: dict = {}
        for photo, label in zip(group, labels):
            clusters.setdefault(int(label), []).append(photo)
        moments.extend(clusters.values())
    return moments
