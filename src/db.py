import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "pdm.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    type TEXT,
    air_temp_k REAL,
    process_temp_k REAL,
    rot_speed_rpm REAL,
    torque_nm REAL,
    tool_wear_min REAL,
    vibration_mms REAL,
    current_a REAL,
    humidity_pct REAL,
    machine_failure INTEGER,
    collected_at TEXT,
    UNIQUE (machine_id, ts)
);

CREATE INDEX IF NOT EXISTS ix_sensor_ts
ON sensor_raw (ts);

CREATE INDEX IF NOT EXISTS ix_sensor_machine
ON sensor_raw (machine_id, ts);

CREATE TABLE IF NOT EXISTS collect_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    window_start TEXT,
    window_end TEXT,
    rows_received INTEGER NOT NULL,
    rows_inserted INTEGER NOT NULL,
    rows_skipped INTEGER NOT NULL,
    note TEXT
);
"""

COLUMNS = [
    "machine_id",
    "ts",
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
    "machine_failure",
    "collected_at",
]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"DB 테이블 준비 완료: {DB_PATH}")


def upsert(con, df):
    df = df.reindex(columns=COLUMNS)  # 저장할 열과 순서 맞춤, 없는 열은 결측값 NULL
    before = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[
        0
    ]  # 저장 전 전체 행 수 확인
    sql = (
        f"INSERT OR IGNORE INTO sensor_raw ({','.join(COLUMNS)}) "  # 제약조건 위반 -> 건너뜀
        f"VALUES ({','.join('?' * len(COLUMNS))})"
    )
    con.executemany(
        sql,
        df.where(pd.notna(df), None).itertuples(
            index=False, name=None
        ),  # 여러 행을 SQL에 전달
    )
    con.commit()  # 변경사항 확정
    after = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    inserted = after - before  # 새로 추가된 행 수 계산
    return inserted, len(df) - inserted  # 추가된 행 수와 건너뛴 행 수 반환


def log_run(con, window_start, window_end, received, inserted, skipped, note=""):
    con.execute(
        "INSERT INTO collect_log (run_at, window_start, window_end,"
        " rows_received, rows_inserted, rows_skipped, note)"
        " VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
        (str(window_start), str(window_end), received, inserted, skipped, note),
    )
    con.commit()


def read_all(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM sensor_raw ORDER BY ts, machine_id", con)
