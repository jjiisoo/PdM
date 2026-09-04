from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F
from clean import run_pipeline


HORIZON = 30
COST_FN = 8_000_000
COST_FP = 300_000
SEED = 42


def load_history() -> pd.DataFrame:
    files = sorted((ROOT / "data" / "history").glob("*.csv"))

    if not files:
        raise SystemExit(
            "data/history/*.csv가 없습니다. collector.py를 먼저 실행하세요."
        )

    return pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True,
    )


def calculate_cost(y_true, probability, threshold):
    pred = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    cost = fn * COST_FN + fp * COST_FP

    return int(cost), pred, (tn, fp, fn, tp)


def main() -> int:
    files = sorted((ROOT / "data" / "history").glob("*.csv"))
    raw = load_history()

    print(f"원본 {len(raw):,}행 ({len(files)}개 파일)")

    clean, log, rep = run_pipeline(raw, verbose=True)
    clean = clean[~clean["is_gap"].astype(bool)].copy()

    d = F.make_horizon_label(
        clean,
        horizon=HORIZON,
    )

    d = F.build(
        d,
        windows=(10, 30, 60),
        shift_one=True,
    )

    feat = [col for col in F.feature_columns(d) if col not in ("y", "machine_failure")]

    d = d.dropna(subset=feat + ["y"]).sort_values("ts").reset_index(drop=True)

    if len(d) < 500:
        raise SystemExit(f"학습 데이터가 부족합니다: {len(d):,}행")

    X = d[feat].to_numpy()
    y = d["y"].to_numpy(dtype=int)

    cut1 = int(len(d) * 0.6)
    cut2 = int(len(d) * 0.8)

    Xtr = X[:cut1]
    ytr = y[:cut1]

    Xval = X[cut1:cut2]
    yval = y[cut1:cut2]

    Xte = X[cut2:]
    yte = y[cut2:]

    if np.unique(ytr).size < 2 or np.unique(yval).size < 2 or np.unique(yte).size < 2:
        raise SystemExit(
            "train, validation, test 중 한 구간에 클래스가 하나만 있습니다."
        )

    print(f"\ntrain {len(Xtr):,} / validation {len(Xval):,} / test {len(Xte):,}")

    print(
        f"양성률 | "
        f"train {ytr.mean() * 100:.2f}% / "
        f"validation {yval.mean() * 100:.2f}% / "
        f"test {yte.mean() * 100:.2f}%"
    )

    mdl = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )

    mdl.fit(
        Xtr,
        ytr,
        eval_set=[(Xval, yval)],
        verbose=False,
    )

    val_prob = mdl.predict_proba(Xval)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 197)
    validation_costs = []

    for threshold in thresholds:
        cost, _, _ = calculate_cost(
            yval,
            val_prob,
            threshold,
        )
        validation_costs.append(cost)

    best_index = int(np.argmin(validation_costs))
    threshold = float(thresholds[best_index])

    test_prob = mdl.predict_proba(Xte)[:, 1]

    test_cost, test_pred, test_cm = calculate_cost(
        yte,
        test_prob,
        threshold,
    )

    cost_05, pred_05, cm_05 = calculate_cost(
        yte,
        test_prob,
        0.5,
    )

    tn, fp, fn, tp = test_cm

    metrics = {
        "n_rows": int(len(d)),
        "n_features": int(len(feat)),
        "horizon_min": HORIZON,
        "train_rows": int(len(Xtr)),
        "validation_rows": int(len(Xval)),
        "test_rows": int(len(Xte)),
        "train_end": str(d["ts"].iloc[cut1 - 1]),
        "validation_start": str(d["ts"].iloc[cut1]),
        "validation_end": str(d["ts"].iloc[cut2 - 1]),
        "test_start": str(d["ts"].iloc[cut2]),
        "positive_rate_train": round(float(ytr.mean()), 4),
        "positive_rate_validation": round(float(yval.mean()), 4),
        "positive_rate_test": round(float(yte.mean()), 4),
        "roc_auc": round(
            float(roc_auc_score(yte, test_prob)),
            4,
        ),
        "pr_auc": round(
            float(average_precision_score(yte, test_prob)),
            4,
        ),
        "threshold": round(threshold, 3),
        "precision": round(
            float(
                precision_score(
                    yte,
                    test_pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    yte,
                    test_pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    yte,
                    test_pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "validation_cost_at_threshold": int(validation_costs[best_index]),
        "test_cost_at_threshold": int(test_cost),
        "test_cost_at_0.5": int(cost_05),
    }

    print("\n[성능]")

    for key, value in metrics.items():
        print(f"  {key:<30} {value}")

    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["실제정상", "실제고장"],
        columns=["예측정상", "예측고장"],
    )

    print("\n[혼동행렬]")
    print(cm.to_string())

    importance = pd.Series(
        mdl.feature_importances_,
        index=feat,
    ).sort_values(ascending=False)

    print("\n[변수 중요도 상위 10]")
    print(importance.head(10).round(4).to_string())

    outdir = ROOT / "models"
    outdir.mkdir(exist_ok=True)

    (outdir / "metrics_xgb.json").write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    importance.head(30).to_csv(
        outdir / "feature_importance_xgb.csv",
        header=["importance"],
    )

    pd.DataFrame(
        {
            "ts": d["ts"].iloc[cut2:].to_numpy(),
            "machine_id": d["machine_id"].iloc[cut2:].to_numpy(),
            "y": yte,
            "prob": test_prob,
            "pred": test_pred,
        }
    ).to_csv(
        outdir / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "threshold": thresholds,
            "validation_cost": validation_costs,
        }
    ).to_csv(
        outdir / "threshold_costs.csv",
        index=False,
    )

    print(
        "\n저장: models/metrics.json, "
        "feature_importance.csv, "
        "test_predictions.csv, "
        "threshold_costs.csv"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
