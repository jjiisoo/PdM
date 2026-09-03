from __future__ import annotations

import numpy as np
import pandas as pd


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


class StepLog:
    def __init__(self):
        self.rows = []

    def __call__(self, name, df):
        prev = self.rows[-1][1] if self.rows else len(df)
        self.rows.append((name, len(df), len(df) - prev))
        return df

    def frame(self):
        return pd.DataFrame(self.rows, columns=["단계", "행수", "증감"])


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


def snap_timestamp(df, freq="min"):
    df = df.copy()
    df["ts"] = df["ts"].dt.round(freq)
    return df


def drop_dups(df):
    df = df.copy()

    if "collected_at" in df.columns:
        df = df.sort_values(
            "collected_at",
            kind="stable",
            na_position="first",
        )

    return (
        df.drop_duplicates(["machine_id", "ts"], keep="last")
        .sort_values(["machine_id", "ts"])
        .reset_index(drop=True)
    )


def detect_and_fix_temp_unit(df, cols=("air_temp_k", "process_temp_k")):
    df = df.copy()
    report = {}

    for c in cols:
        lo, hi = PHYS_RANGE[c]
        converted = df[c] + 273.15
        mask = df[c].notna() & df[c].lt(200) & df[c].ne(0) & converted.between(lo, hi)

        report[c] = int(mask.sum())
        df.loc[mask, c] = converted.loc[mask]

    return df, report


def detect_vibration_unit(
    df: pd.DataFrame,
    col="vibration_mms",
    factor=9.81,
    ratio=4.0,
):
    df = df.copy()
    med = df.groupby("machine_id")[col].transform("median")
    mask = df[col].notna() & med.gt(0) & df[col].gt(med * ratio)
    df.loc[mask, col] = df.loc[mask, col] / factor
    return df, int(mask.sum())


def range_check(df, rng=None):
    ranges = PHYS_RANGE if rng is None else rng
    df = df.copy()
    report = {}

    for c, (lo, hi) in ranges.items():
        bad = df[c].notna() & ~df[c].between(lo, hi)
        report[c] = int(bad.sum())
        df.loc[bad, c] = np.nan

    return df, report


def hampel_flag(s, window=11, n_sigma=5.0, center=False):
    if window < 3 or n_sigma <= 0:
        raise ValueError("window는 3 이상, n_sigma는 양수여야 합니다.")

    rolling = s.rolling(window, center=center, min_periods=3)
    med = rolling.median()

    def window_mad(values):
        middle = np.nanmedian(values)
        return np.nanmedian(np.abs(values - middle))

    mad = rolling.apply(window_mad, raw=True)
    sigma = (1.4826 * mad).replace(0, np.nan)

    return ((s - med).abs() > n_sigma * sigma).fillna(False)


def flag_spikes(df: pd.DataFrame, cols=None, window=11, n_sigma=5.0):
    cols = SENSOR_COLS if cols is None else list(cols)
    df = df.sort_values(["machine_id", "ts"]).reset_index(drop=True).copy()

    continuous = (
        df["machine_id"].ne(df["machine_id"].shift())
        | df["ts"].diff().ne(pd.Timedelta(minutes=1))
    ).cumsum()

    spike_cols = []

    for c in cols:
        if c not in df.columns:
            continue

        flag_col = f"spike_{c}"
        df[flag_col] = df.groupby(continuous)[c].transform(
            lambda s: hampel_flag(s, window, n_sigma)
        )
        spike_cols.append(flag_col)

    df["spike_any"] = df[spike_cols].any(axis=1)
    df["spike_count"] = df[spike_cols].sum(axis=1)

    return df


def interpolate_short_gaps(df, cols=None, max_gap=5):
    if max_gap < 1:
        raise ValueError("max_gap은 1 이상이어야 합니다.")

    cols = SENSOR_COLS if cols is None else list(cols)
    df = df.sort_values(["machine_id", "ts"]).reset_index(drop=True).copy()
    filled = {c: 0 for c in cols}

    if df["ts"].isna().any():
        raise ValueError("ts에 결측값이 있습니다.")

    if df.duplicated(["machine_id", "ts"]).any():
        raise ValueError("보간 전에 drop_dups를 실행하세요.")

    continuous = (
        df["machine_id"].ne(df["machine_id"].shift())
        | df["ts"].diff().ne(pd.Timedelta(minutes=1))
    ).cumsum()

    for _, group in df.groupby(continuous):
        for c in cols:
            s = pd.Series(
                group[c].to_numpy(dtype=float, na_value=np.nan),
                index=pd.DatetimeIndex(group["ts"]),
            )

            missing = s.isna()
            block = missing.ne(missing.shift()).cumsum()
            gap_size = missing.groupby(block).transform("sum")
            eligible = missing & gap_size.le(max_gap)

            flag_col = f"spike_{c}"
            if flag_col in group.columns:
                flagged = group[flag_col].eq(True).to_numpy()
                anchor = pd.Series(
                    s.notna().to_numpy() & ~flagged,
                    index=s.index,
                )
                eligible &= (
                    anchor.ffill()
                    if False
                    else (
                        anchor.where(s.notna()).ffill().eq(True)
                        & anchor.where(s.notna()).bfill().eq(True)
                    )
                )

            candidate = s.interpolate(method="time", limit_area="inside")
            fixed = s.copy()
            fixed.loc[eligible] = candidate.loc[eligible]

            filled[c] += int((missing & fixed.notna()).sum())
            df.loc[group.index, c] = fixed.to_numpy()

    return df, filled


def estimate_drift(df, col="process_temp_k", ref="air_temp_k"):
    out = {machine_id: np.nan for machine_id in df["machine_id"].dropna().unique()}

    valid = df[["ts", "machine_id", col, ref]].notna().all(axis=1)

    for c in (col, ref):
        flag_col = f"spike_{c}"
        if flag_col in df.columns:
            valid &= ~df[flag_col].eq(True)

    data = df.loc[valid].copy()

    if data.empty:
        daily = pd.DataFrame(columns=["machine_id", "d", "v", "fleet", "resid"])
        return out, daily

    origin = df["ts"].min().floor("D")
    data["diff"] = data[col] - data[ref]
    data["d"] = (data["ts"].dt.floor("D") - origin).dt.days

    daily = data.groupby(["machine_id", "d"])["diff"].median().rename("v").reset_index()

    fleet = daily.groupby("d")["v"].median().rename("fleet")
    daily = daily.join(fleet, on="d")
    daily["resid"] = daily["v"] - daily["fleet"]

    for machine_id, group in daily.groupby("machine_id"):
        if group["d"].nunique() < 2:
            continue

        slope = np.polyfit(group["d"], group["resid"], 1)[0]
        out[machine_id] = float(slope)

    return out, daily


def correct_drift(
    df: pd.DataFrame,
    slopes: dict,
    col="process_temp_k",
    min_slope: float = 0.05,
):
    df = df.copy()
    applied = {}

    if df.empty:
        return df, applied

    t0 = df["ts"].min().floor("D")
    days = (df["ts"] - t0).dt.total_seconds() / 86400.0

    for machine_id, slope in slopes.items():
        if not np.isfinite(slope) or abs(slope) < min_slope:
            applied[machine_id] = 0.0
            continue

        mask = df["machine_id"] == machine_id
        df.loc[mask, col] = df.loc[mask, col] - slope * days.loc[mask]
        applied[machine_id] = float(slope)

    return df, applied


def reindex_time(df, freq="min"):
    if df.empty:
        out = df.copy()
        out["is_gap"] = pd.Series(index=out.index, dtype=bool)
        return out

    parts = []

    for machine_id, group in df.groupby("machine_id"):
        group = group.set_index("ts").sort_index()

        if not group.index.is_unique:
            raise ValueError("reindex_time 전에 drop_dups를 실행하세요.")

        full = pd.date_range(
            group.index.min(),
            group.index.max(),
            freq=freq,
        )
        added = ~full.isin(group.index)
        out = group.reindex(full)

        if "is_gap" in out.columns:
            previous_gap = out["is_gap"].eq(True).to_numpy()
            out["is_gap"] = added | previous_gap
        else:
            out["is_gap"] = added

        out["machine_id"] = machine_id
        out.index.name = "ts"
        parts.append(out.reset_index())

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["ts", "machine_id"])
        .reset_index(drop=True)
    )


def run_pipeline(raw: pd.DataFrame, verbose: bool = True):
    log = StepLog()
    rep = {}

    df = log("0. 원본 수신", raw.copy())
    df = log("1. 타입 강제", coerce_types(df))
    df = log("2. 타임스탬프 스냅", snap_timestamp(df))
    df = log("3. 중복 제거", drop_dups(df))

    df, rep["temp_unit"] = detect_and_fix_temp_unit(df)
    df = log("4a. 온도 단위 통일", df)

    df, rep["vib_unit"] = detect_vibration_unit(df)
    df = log("4b. 진동 단위 통일", df)

    df, rep["range"] = range_check(df)
    df = log("5. 물리범위 → NaN", df)

    df = flag_spikes(df)
    df = log("6. 스파이크 플래그", df)

    df, rep["filled"] = interpolate_short_gaps(df)
    df = log("7. 짧은 결측 보간", df)

    slopes, rep["drift_daily"] = estimate_drift(df)
    rep["drift_slopes"] = slopes

    df, rep["drift_applied"] = correct_drift(df, slopes)
    df = log("8. 드리프트 보정", df)

    df = reindex_time(df)
    df = log("9. 시간축 재색인", df)

    if verbose:
        print(log.frame().to_string(index=False))

    return df, log, rep
