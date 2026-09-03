import numpy as np


def score(pred, gt, name):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    if pred.shape != gt.shape:
        raise ValueError("pred와 gt의 모양이 같아야 합니다.")

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    return {
        "규칙": name,
        "실제": int(gt.sum()),
        "탐지": int(pred.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "정밀도": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "재현율": round(tp / (tp + fn), 3) if tp + fn else 0.0,
    }
