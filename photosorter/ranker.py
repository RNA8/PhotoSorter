from dataclasses import dataclass


@dataclass
class PhotoScore:
    photo_id: int
    gaze: float
    smile: float
    eyes: float
    sharpness: float
    exposure: float
    composite: float
    rank: int


def rank_moment(photo_scores: list, weights, keep_threshold: float) -> list:
    computed = []
    for ps in photo_scores:
        sharpness = (ps.get('sharpness', 0.0) + ps.get('blur_score', 0.0)) / 2.0
        composite = (
            ps['gaze_score'] * weights.gaze
            + ps['smile_score'] * weights.smile
            + ps['eyes_score'] * weights.eyes
            + sharpness * weights.sharpness
            + ps['exposure'] * weights.exposure
        )
        computed.append({
            'photo_id': ps['photo_id'],
            'gaze': ps['gaze_score'],
            'smile': ps['smile_score'],
            'eyes': ps['eyes_score'],
            'sharpness': sharpness,
            'exposure': ps['exposure'],
            'composite': float(composite),
        })

    computed.sort(key=lambda x: x['composite'], reverse=True)

    return [
        PhotoScore(**c, rank=rank)
        for rank, c in enumerate(computed, start=1)
    ]


def suggested_keep_ids(ranked: list, keep_threshold: float) -> list:
    if not ranked:
        return []
    threshold = ranked[0].composite * keep_threshold
    return [ps.photo_id for ps in ranked if ps.composite >= threshold]
