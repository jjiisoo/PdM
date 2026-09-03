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
