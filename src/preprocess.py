from __future__ import annotations

import numpy as np
import pandas as pd

# 전처리 적용할 센서 열 목록
SENSOR_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
]

# 센서별 허용 범위
PHYS_RANGE = {
    "air_temp_k": (270.0, 340.0),
    "process_temp_k": (280.0, 360.0),
    "rot_speed_rpm": (500.0, 4000.0),
    "torque_nm": (1.0, 100.0),
    "tool_wear_min": (0.0, 400.0),
    "vibration_mms": (0.05, 30.0),
    "current_a": (0.1, 40.0),
    "humidity_pct": (0.0, 100.0),
}


# 처리 단계마다 행 수와 증감 기록
class StepLog:
    def __init__(self):
        self.rows = []

    def __call__(self, name, df):
        prev = self.rows[-1][1] if self.rows else len(df)
        self.rows.append((name, len(df), len(df) - prev))
        return df

    def frame(self):
        return pd.DataFrame(self.rows, columns=["단계", "행수", "증감"])


# 시각, 숫자 자료형 변환, 시각이나 설비 ID 없는 행 제거
def coerce_types(df):
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df["machine_id"] = df["machine_id"].astype("string").str.strip()
    df["machine_id"] = df["machine_id"].replace("", pd.NA)

    if "collected_at" in df.columns:
        df["collected_at"] = pd.to_datetime(
            df["collected_at"], errors="coerce", utc=True
        )

    for c in SENSOR_COLS + ["machine_failure"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df.dropna(subset=["ts", "machine_id"]).reset_index(drop=True)


# 시각 -> 가장 가까운 분으로 반올림
def snap_timestamp(df, freq="min"):
    df = df.copy()
    df["ts"] = df["ts"].dt.round(freq)
    return df


# 같은 설비, 같은 시각의 기록 중 마지막 수집 기록 유지
def drop_dups(df):
    df = df.copy()

    if "collected_at" in df.columns:
        df = df.sort_values("collected_at", kind="stable", na_position="first")
    return (
        df.drop_duplicates(subset=["machine_id", "ts"], keep="last")
        .sort_values(["machine_id", "ts"])
        .reset_index(drop=True)
    )


# 200 미만 온도를 섭씨로 가정 -> k로 변환
def detect_and_fix_temp_unit(df, cols=("air_temp_k", "process_temp_k")):
    df = df.copy()
    report = {}
    for c in cols:
        mask = df[c].notna() & (df[c] < 200)
        report[c] = int(mask.sum())
        df.loc[mask, c] = df.loc[mask, c] + 273.15
    return df, report


# 허용 범위 확인, 밖의 센서값 -> NaN
def range_check(df, rng=None):
    rng = rng or PHYS_RANGE
    df = df.copy()
    report = {}
    for c, (lo, hi) in rng.items():
        bad = df[c].notna() & ~df[c].between(lo, hi)
        report[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    return df, report


# 주변 중앙값, 편차 이용해 이상값 위치 표시
def hampel_flag(s, window=11, n_sigma=5.0):
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    sigma = 1.4826 * mad
    sigma = sigma.replace(0, np.nan)
    return ((s - med).abs() > n_sigma * sigma).fillna(False)


# 결측 센서값 보간
def interpolate_short_gaps(df, cols=None, max_gap=5):
    """max_gap분 이하의 짧은 구간만 시간 보간합니다."""
    df = df.sort_values(["machine_id", "ts"]).copy()
    filled = {}
    for c in cols:
        before = df[c].isna().sum()
        df[c] = df.groupby("machine_id")[c].transform(
            lambda s: s.interpolate(
                method="linear", limit=max_gap, limit_direction="both"
            )
        )

        filled[c] = int(before - df[c].isna().sum())
    return df, filled


# 다른 설비들과 비교해 온도 차이가 서서히 변하는 정도 추정
def estimate_drift(df, col="process_temp_k", ref="air_temp_k"):
    d = df.dropna(subset=[col, ref]).copy()
    d["diff"] = d[col] - d[ref]
    d["day"] = (d["ts"] - d["ts"].min()).dt.total_seconds() / 86400.0
    daily = (
        d.groupby(["machine_id", d["day"].astype(int)])["diff"]
        .median()
        .rename("v")
        .reset_index()
        .rename(columns={"day": "d"})
    )

    fleet = daily.groupby("d")["v"].median().rename("fleet")
    daily = daily.join(fleet, on="d")
    daily["resid"] = daily["v"] - daily["fleet"]  # 라인 대비 잔차
    out = {}
    for m, g in daily.groupby("machine_id"):
        slope = np.polyfit(g["d"], g["resid"], 1)[0]
        out[m] = float(slope)
    return out, daily


def reindex_time(df, freq="min"):
    parts = []
    for m, g in df.groupby("machine_id"):
        g = g.set_index("ts").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        g2 = g.reindex(full)
        g2["is_gap"] = g2["machine_id"].isna()
        g2["machine_id"] = m
        g2.index.name = "ts"
        parts.append(g2.reset_index())
    return pd.concat(parts, ignore_index=True).sort_values(["ts", "machine_id"])
