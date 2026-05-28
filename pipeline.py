import argparse
import torch
import open_clip
from insightface.app import FaceAnalysis

from photosorter.config import load_config
from photosorter.db import (
    init_db, insert_photo, insert_moment, insert_moment_photos, insert_scores,
)
from photosorter.ingestor import scan_folder
from photosorter.grouper import group_photos
from photosorter.scorer import score_image_quality, normalize_sharpness, score_faces
from photosorter.ranker import rank_moment, suggested_keep_ids


def main():
    parser = argparse.ArgumentParser(description='PhotoSorter pipeline')
    parser.add_argument('--input', required=True, help='Folder of photos to process')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = init_db(cfg.db_path)

    print("Scanning folder...")
    photos = scan_folder(args.input)
    for p in photos:
        pid = insert_photo(conn, p['path'], p['timestamp'],
                           p['gps_lat'], p['gps_lon'], p['exif_json'])
        p['id'] = pid
    print(f"Found {len(photos)} photos")

    if not photos:
        print("No photos found. Exiting.")
        return

    print("Loading CLIP model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.clip_model, pretrained=cfg.clip_pretrained
    )
    model = model.to(device).eval()

    print("Grouping photos into moments...")
    moments = group_photos(
        photos, cfg.time_window_minutes, cfg.dbscan_eps,
        cfg.dbscan_min_samples, model, preprocess, device,
    )
    print(f"Found {len(moments)} moments")

    print("Loading InsightFace model...")
    face_analyzer = FaceAnalysis(
        name=cfg.insightface_model,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    face_analyzer.prepare(ctx_id=0, det_size=tuple(cfg.det_size))

    print("Scoring and ranking photos...")
    for i, moment_photos in enumerate(moments):
        print(f"  Moment {i + 1}/{len(moments)} ({len(moment_photos)} photos)", end='\r')

        quality_scores = []
        face_score_list = []
        for photo in moment_photos:
            qs = score_image_quality(photo['path'])
            fs = score_faces(photo['path'], face_analyzer)
            quality_scores.append({**qs, 'photo_id': photo['id']})
            face_score_list.append({**fs, 'photo_id': photo['id']})

        quality_scores = normalize_sharpness(quality_scores)

        combined = [{**qs, **fs} for qs, fs in zip(quality_scores, face_score_list)]
        ranked = rank_moment(combined, cfg.weights, cfg.keep_threshold)

        timestamps = [p['timestamp'] for p in moment_photos]
        moment_id = insert_moment(conn, min(timestamps), max(timestamps), len(moment_photos))
        insert_moment_photos(conn, moment_id, [(ps.photo_id, ps.rank) for ps in ranked])
        for ps in ranked:
            insert_scores(conn, ps.photo_id, ps.gaze, ps.smile, ps.eyes,
                          ps.sharpness, ps.exposure, ps.composite)

    print(f"\nDone. {len(moments)} moments stored in {cfg.db_path}")
    print(f"Start the review server: uvicorn photosorter.api:app --port {cfg.port}")


if __name__ == '__main__':
    main()
