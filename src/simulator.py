from __future__ import annotations

import numpy as np
import pandas as pd


MACHINES = {
    # machine_id : (품질등급, 마모한계 계수(과부하 고장 판단 기준), 공구교체주기(분))
    "CNC-01": {"type": "L", "osf_limit": 11000, "tool_life": 210},
    "CNC-02": {"type": "M", "osf_limit": 12000, "tool_life": 225},
    "CNC-03": {"type": "H", "osf_limit": 13000, "tool_life": 240},
}

POLLUTION = {
    "dropout_rate": 0.015,  # 통신 끊김으로 통째로 사라지는 구간 발생 확률
    "dropout_len": (3, 40),  # 끊김 길이(분)
    "nan_rate": 0.008,  # 개별 센서값만 NaN 0.8 확률로 결측치 넣겠다.
    "spike_rate": 0.004,  # 센서 튐(전기 노이즈)
    "dup_rate": 0.006,  # 같은 레코드 중복 전송
    "ts_jitter_rate": 0.05,  # 타임스탬프 흔들림
    "unit_mix_rate": 0.10,  # 단위 혼재(K 대신 섭씨로 오는 구간)
    "drift_per_day": 0.35,  # 온도 센서 드리프트 (K/day)
}


def _simulate_one(  # 어느 설비 / 몇 분 동안의 데이터 / 시작 시각 / 난수 만들어주는 객체
    machine_id: str, n_minutes: int, start: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    spec = MACHINES[machine_id]  # 설비 설정
    ts = pd.date_range(
        start, periods=n_minutes, freq="min"
    )  # 시간 목록 (분 간격으로 새 시점 생김)

    # ---- 공정 부하
    hour = ts.hour + ts.minute / 60.0  # ex. 9시 30 : 9.5
    duty = (
        0.55 + 0.45 * np.sin((hour - 6) / 24 * 2 * np.pi)
    )  # 0.1 ~ 1.0 # 주기적으로 오르내리는 값 생성 -> 시간대에 따라 부하가 높아졌다 낮아졌다
    duty = np.clip(duty + rng.normal(0, 0.05, n_minutes), 0.05, 1.0)

    # ---- 공기 온도
    air = 298.0 + 2.0 * np.sin((hour - 14) / 24 * 2 * np.pi)
    air = (
        air + np.cumsum(rng.normal(0, 0.02, n_minutes))
    )  # cumsum() -> 누적합 / 각 시점의 작은 변화 누적 -> 온도가 시간에 따라 떠다니는 형태
    air = air + rng.normal(0, 0.15, n_minutes)

    # ---- 공구 마모, 누적되다가 교체하면 0
    tool_life = spec["tool_life"]
    wear_rate = 1.0 + 0.6 * duty  # 부하 클수록 마모 빠르게 쌓이도록
    wear = np.zeros(n_minutes)
    acc = rng.uniform(0, 60)

    limit = tool_life * rng.uniform(0.90, 1.15)
    for i in range(n_minutes):
        acc += wear_rate[i]
        if acc > limit:  # 계획 교체 (정비반 재량으로 조금씩 다름)
            acc = 0.0
            limit = tool_life * rng.uniform(0.90, 1.15)
        wear[i] = acc

    # --- 회전수: 부하가 클수록 회전수 낮아짐
    rpm = 2860 - 1500 * duty + rng.normal(0, 45, n_minutes)
    rpm = np.clip(rpm, 1150, 2900)

    # --- 토크: 부하가 클수록 토크 커짐
    torque = 10 + 40 * duty + 0.02 * wear + rng.normal(0, 2.0, n_minutes)
    torque = np.clip(torque, 3.0, 80.0)

    hvac_fail = np.zeros(n_minutes, dtype=bool)
    for _ in range(max(1, n_minutes // 2000)):
        s = rng.integers(0, max(1, n_minutes - 120))
        hvac_fail[s : s + rng.integers(40, 120)] = True
    air = air + 5.5 * hvac_fail  # 실내 온도 상승

    # --- 토크와 회전수로 전력 계산
    power_w = torque * rpm * 2 * np.pi / 60.0
    # proc : 공기 온도, 전력, 마모, 냉각 상태로 공정 온도 계산
    proc = air + 8.5 + power_w / 1400.0 + 0.004 * wear
    proc = proc - 6.0 * hvac_fail
    proc = proc + rng.normal(0, 0.12, n_minutes)

    # --- 진동: 회전수와 마모에 따라 진동 커짐
    vib = (
        0.8
        + 0.0009 * rpm
        + 0.9 * (wear / tool_life) ** 3
        + rng.normal(0, 0.06, n_minutes)
    )
    vib = np.clip(vib, 0.1, None)

    # --- 전류: 전력 바탕으로 전류 계산
    current = power_w / (380 * 1.732 * 0.85) + rng.normal(0, 0.15, n_minutes)
    current = np.clip(current, 0.2, None)

    # --- 습도: 온도 높을수록 습도 낮아짐
    humid = 55 - 1.8 * (air - 298) + rng.normal(0, 2.5, n_minutes)
    humid = np.clip(humid, 15, 95)

    df = pd.DataFrame(
        {
            "ts": ts,
            "machine_id": machine_id,
            "type": spec["type"],
            "air_temp_k": air,
            "process_temp_k": proc,
            "rot_speed_rpm": rpm,
            "torque_nm": torque,
            "tool_wear_min": wear,
            "vibration_mms": vib,
            "current_a": current,
            "humidity_pct": humid,
        }
    )

    # ------------------------------------------------------------------
    # 고장 라벨
    # ------------------------------------------------------------------
    twf = (
        (wear >= 200) & (wear <= 240) & (rng.random(n_minutes) < 0.004)
    )  # 공구 마모 고장
    hdf = ((proc - air) < 8.6) & (rpm < 1380)  # 방열 고장
    pwf = (power_w < 3500) | (power_w > 9000)  # 전력 고장
    osf = (wear * torque) > spec["osf_limit"]  # 과도한 부하에 따른 고장
    rnf = rng.random(n_minutes) < 0.0002  # 무작위 고장

    df["twf"] = twf.astype(int)
    df["hdf"] = hdf.astype(int)
    df["pwf"] = pwf.astype(int)
    df["osf"] = osf.astype(int)
    df["rnf"] = rnf.astype(int)
    df["machine_failure"] = (twf | hdf | pwf | osf | rnf).astype(int)
    df["power_w"] = power_w
    return df


def simulate_truth(  # 설비 세 대의 결과 합치기
    n_minutes: int = 1440, start: str | pd.Timestamp = "2024-01-01", seed: int = 42
) -> pd.DataFrame:
    """오염 없는 참값을 생성합니다."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)

    parts = []
    for m in MACHINES:  # 각 설비에 대해 함수 실행
        result = _simulate_one(m, n_minutes, start, rng)
        parts.append(result)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["ts", "machine_id"]).reset_index(drop=True)


truth = simulate_truth(
    n_minutes=1440 * 14, start="2024-01-01", seed=42
)  # 하루는 1440min, 14일 데이터
print("설비 수 :", truth["machine_id"].nunique())
print("기간 :", truth["ts"].min(), "~", truth["ts"].max())
print("행 수 :", f"{len(truth):,}")

modes = truth[["twf", "hdf", "pwf", "osf", "rnf", "machine_failure"]].sum()
print(pd.DataFrame({"건수": modes, "비율(%)": (modes / len(truth) * 100).round(3)}))

cols = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "power_w",
]
print(truth[cols].describe().loc[["mean", "std", "min", "50%", "max"]].round(2))
# ----------------------------------------------------------------------
# 2) 현장급 오염 주입
# ----------------------------------------------------------------------
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


# 오염 데이터 주입
def pollute(
    truth: pd.DataFrame,
    seed: int = 7,
    cfg: dict | None = None,
    return_masks: bool = False,
):
    c = dict(POLLUTION)
    if cfg:
        c.update(cfg)
    rng = np.random.default_rng(seed)
    df = truth.copy()
    masks = pd.DataFrame(index=df.index)

    df = df.drop(columns=["twf", "hdf", "pwf", "osf", "rnf", "power_w"])

    t0 = df["ts"].min()
    days = (df["ts"] - t0).dt.total_seconds() / 86400.0

    # ---- 센서 드리프트: CNC-02 공정 온도에 시간에 비례하는 값 추가, 시간 갈수록 온도가 실제보다 높게 측정됨
    m2 = df["machine_id"] == "CNC-02"
    df.loc[m2, "process_temp_k"] += c["drift_per_day"] * days[m2]

    # ---- 단위 혼재: 같은 열에 K / C 섞임
    n = len(df)
    unit_block = np.zeros(n, dtype=bool)
    n_blocks = max(1, int(n * c["unit_mix_rate"] / 200))
    for _ in range(n_blocks):
        s = rng.integers(0, n - 200)
        unit_block[s : s + 200] = True
    df.loc[unit_block, "air_temp_k"] -= 273.15
    df.loc[unit_block, "process_temp_k"] -= 273.15
    masks["unit_temp"] = unit_block
    # 진동 단위도 일부는 m/s^2 로 (×9.81)
    vib_block = rng.random(n) < 0.04
    df.loc[vib_block, "vibration_mms"] *= 9.81
    masks["unit_vib"] = vib_block

    # ---- 센서 튐: 일부 값을 크게 곱하거나 0으로 변경 / 값자기 큰 값 또는 0이 나타남
    for col in SENSOR_COLS:
        hit = rng.random(n) < c["spike_rate"]
        mode = rng.random(n)
        df.loc[hit & (mode < 0.5), col] = df.loc[hit & (mode < 0.5), col] * rng.uniform(
            8, 40
        )
        df.loc[hit & (mode >= 0.5), col] = (
            0.0  # 0으로 떨어지는 것도 스파이크 (배선이 빠지거나 신호 유실 -> 0)
        )
        masks[f"spike_{col}"] = hit

    # ---- 개별 결측: 일부 센서값을 NaN으로 변경 / 행은 있지만 특정 값 없
    for col in SENSOR_COLS:
        hit = rng.random(n) < c["nan_rate"]
        df.loc[hit, col] = np.nan
        masks[f"nan_{col}"] = hit

    # ---- 통신 끊김: 연속된 행 삭제 / 해당 시간의 기록 자체가 없음
    drop_mask = np.zeros(n, dtype=bool)
    n_drop = int(n * c["dropout_rate"] / 10)
    for _ in range(max(1, n_drop)):
        s = rng.integers(0, n)
        ln = rng.integers(*c["dropout_len"]) * 3  # 설비 3대 × 분
        drop_mask[s : s + ln] = True
    masks["dropped"] = drop_mask
    keep = ~drop_mask
    df = df[keep].copy()
    kept_masks = masks[keep].copy()

    # ---- 중복 전송: 일부 행 복사해서 추가 / 같은 기록이 여러 번 들어옴
    n2 = len(df)
    dup_idx = rng.random(n2) < c["dup_rate"]
    dups = df[dup_idx].copy()
    kept_masks["is_dup"] = False
    dup_masks = kept_masks[dup_idx].copy()
    dup_masks["is_dup"] = True
    df = pd.concat([df, dups], ignore_index=True)
    kept_masks = pd.concat([kept_masks, dup_masks], ignore_index=True)

    # ---- 시간 흔들림: 시각에 임의의 초를 더함 / 1분 간격에서 벗어남 + 순서 변경: 시간순으로 정렬 안됨
    n3 = len(df)
    jitter = np.where(
        rng.random(n3) < c["ts_jitter_rate"], rng.integers(-90, 90, n3), 0
    )
    df["ts"] = df["ts"] + pd.to_timedelta(jitter, unit="s")
    kept_masks["ts_jittered"] = jitter != 0
    order = rng.permutation(n3)
    df = df.iloc[order].reset_index(drop=True)
    kept_masks = kept_masks.iloc[order].reset_index(drop=True)

    # --- (h) 실제 수집기가 붙이는 메타 컬럼 ---
    df["collected_at"] = pd.Timestamp("2024-01-01")
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")  # 문자열로 들어옴(현실)
    if return_masks:
        return df, kept_masks
    return df


obs, masks = pollute(
    truth, seed=7, return_masks=True
)  # 데이터 오염시키고 오염 위치 기록
print("참값 행수 :", f"{len(truth):,}")
print("관측 행수 :", f"{len(obs):,}", f"({len(obs) - len(truth):+,})")

print(obs.head(3).to_string())
print(
    (obs[SENSOR_COLS].isna().mean() * 100).round(2).to_string()
)  # to_string -> 출력 형식 제어, 깔끔하게 고정

print(obs["air_temp_k"].describe().round(2).to_string())
print("200 K 미만 비율: %.2f%%" % ((obs["air_temp_k"] < 200).mean() * 100))

inj = pd.DataFrame({"건수": masks.sum(), "비율(%)": (masks.mean() * 100).round(3)})
print(inj.to_string())


# ----------------------------------------------------------------------
# 3) 실시간 수집용: "지금부터 n분 치"
# ----------------------------------------------------------------------
def sample_window(
    n_minutes: int = 60, end: pd.Timestamp | None = None, seed: int | None = None
) -> pd.DataFrame:
    """수집기가 호출하는 함수. 최근 n분 구간의 관측 데이터를 돌려줍니다."""
    end = pd.Timestamp.utcnow().floor("min") if end is None else pd.Timestamp(end)
    start = end - pd.Timedelta(minutes=n_minutes)
    # 시드를 날짜에서 뽑으면 같은 날 다시 돌려도 같은 값이 나옵니다(재현성)
    if seed is None:
        seed = int(start.strftime("%Y%m%d%H"))
    truth = simulate_truth(n_minutes=n_minutes, start=start, seed=seed)
    obs = pollute(truth, seed=seed + 1)
    obs["collected_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return obs


if __name__ == "__main__":
    t = simulate_truth(n_minutes=1440, start="2024-01-01", seed=42)
    o = pollute(t, seed=7)
    print("truth   :", t.shape)
    print("observed:", o.shape)
    print(o.head(3).to_string())


# '있어야 할 행이 다 있는지' 확인 / 결측률과 데이터 손실률은 다르다
# 0도 스파이크가 될 수 있음
# 단위 통일 (스케일링) - > Standardization, Normalization


print(truth.head())  # 기준 데이터
print(obs.head())  # 오류가 섞인 데이터
print(truth.shape)
print(obs.shape)


truth.to_csv("설비_참값.csv", index=False, encoding="utf-8-sig")
obs.to_csv("설비_관측.csv", index=False, encoding="utf-8-sig")


def sample_window(n_minutes=60, end=None, seed=None):
    end = pd.Timestamp.utcnow().floor("min") if end is None else pd.Timestamp(end)
    start = end - pd.Timedelta(minutes=n_minutes)
    # ★ 시드를 시각에서 뽑으면 같은 날 다시 돌려도 같은 값이 나옵니다
    if seed is None:
        seed = int(start.strftime("%Y%m%d%H"))
    truth = simulate_truth(n_minutes=n_minutes, start=start, seed=seed)
    obs = pollute(truth, seed=seed + 1)
    return obs
