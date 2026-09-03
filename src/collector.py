import argparse
import importlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

import db as dbmod


ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "history"


def fetch(minutes, end=None, source="simulator"):  # simulator에서 데이터 가져옴
    module = importlib.import_module(source)
    return module.sample_window(n_minutes=minutes, end=end)


def save_csv(raw, history_dir):  # 날짜별 CSV에 저장, 중복 안되게, 있어도 되고 없어도 됨
    history_dir.mkdir(parents=True, exist_ok=True)
    filenames = []

    for day, batch in raw.groupby(raw["ts"].str[:10]):
        csv_path = history_dir / f"{day}.csv"

        if csv_path.exists():
            old = pd.read_csv(csv_path)
            batch = pd.concat([old, batch], ignore_index=True)

        batch = batch.drop_duplicates(  # 중복 방지, 첫번째 값 유지
            subset=["machine_id", "ts"],
            keep="first",
        )
        batch = batch.sort_values(["ts", "machine_id"])

        temporary = csv_path.with_suffix(".csv.tmp")  # 기존 파일 보존하면서 저장
        batch.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(csv_path)

        filenames.append(csv_path.name)

    return filenames


def main():
    ap = argparse.ArgumentParser(description="설비 센서 데이터 수집기")
    ap.add_argument("--minutes", type=int, default=1440)
    ap.add_argument("--end", default=None)
    ap.add_argument("--db", type=Path, default=ROOT / "pdm.sqlite3")
    ap.add_argument("--hist", type=Path, default=HIST)
    ap.add_argument("--source", default="simulator")
    args = ap.parse_args()

    if args.minutes <= 0:
        ap.error("--minutes는 양수여야 합니다.")

    # 1. 데이터 수집
    try:
        raw = fetch(args.minutes, args.end, args.source)
    except Exception as e:
        print(f"[ERROR] 수집 실패: {type(e).__name__}: {e}")
        return 1

    if raw.empty:
        print("[WARN] 받은 데이터가 0건입니다.")
        return 0

    try:
        # 2. 저장할 열 확인
        missing = set(dbmod.COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError(f"필수 열 누락: {sorted(missing)}")

        raw = raw.loc[:, dbmod.COLUMNS].copy()

        if raw[["machine_id", "ts"]].isna().any().any():
            raise ValueError("machine_id와 ts에는 결측값이 없어야 합니다.")

        # 시간대 없는 시각은 UTC로 가정
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="raise").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        if raw["ts"].isna().any():
            raise ValueError("유효하지 않은 ts가 있습니다.")

        run_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
        raw["collected_at"] = run_at

        received = len(raw)
        w_start = raw["ts"].min()
        w_end = raw["ts"].max()

        # 3. DB 연결 및 테이블 준비
        args.db.parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(args.db, timeout=30)) as con:
            # SCHEMA의 collect_log (...)는 실제 컬럼 정의로 완성해야 합니다.
            con.executescript(dbmod.SCHEMA)

            # 4. 날짜별 CSV 저장
            csv_names = save_csv(raw, args.hist)

            # 5. 이번에 받은 데이터만 DB에 적재
            # 결측값은 SQLite의 NULL로 저장
            records = raw.astype(object).where(pd.notna(raw), None)
            inserted, skipped = dbmod.upsert(con, records)

            # 6. 수집 이력 기록
            con.execute(
                """
                INSERT INTO collect_log (
                    run_at,
                    window_start,
                    window_end,
                    rows_received,
                    rows_inserted,
                    rows_skipped,
                    note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_at,
                    w_start,
                    w_end,
                    received,
                    inserted,
                    skipped,
                    "csv=" + ",".join(csv_names),
                ),
            )
            con.commit()

    except Exception as e:
        print(f"[ERROR] 저장 실패: {type(e).__name__}: {e}")
        print("일부 저장되었을 수 있으므로 CSV와 DB를 확인하세요.")
        return 1

    print(f"[OK] 수신: {received:,}행")
    print(f"[OK] 삽입: {inserted:,}행")
    print(f"[OK] 건너뜀: {skipped:,}행")
    print(f"[DB] {args.db.resolve()}")
    print(f"[CSV] {args.hist.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# def sample_window(n_minutes=60, end=None, seed=None):
#     end = pd.Timestamp.utcnow().floor("min") if end is None else pd.Timestamp(end)
#     start = end - pd.Timedelta(minutes=n_minutes)
#     # ★ 시드를 시각에서 뽑으면 같은 날 다시 돌려도 같은 값이 나옵니다
#     if seed is None:
#         seed = int(start.strftime("%Y%m%d%H"))
#     truth = simulate_truth(n_minutes=n_minutes, start=start, seed=seed)
#     obs = pollute(truth, seed=seed + 1)
#     return obs
