# collector.py -> 날짜 바꿔가면서 14번 실행하는 코드, 한번에 1일 => 총 14일치

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COLLECTOR = ROOT / "src" / "collector.py"


def main():
    first_end = datetime(2024, 1, 2)  # 첫 수집 종료 시각

    for i in range(14):
        end = first_end + timedelta(days=i)  # 종료시각 하루씩 옮기기
        end_text = end.strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{i + 1}/14] 종료 시각: {end_text}", flush=True)

        subprocess.run(
            [
                sys.executable,
                str(COLLECTOR),
                "--minutes",
                "1440",
                "--end",
                end_text,
            ],
            cwd=ROOT,
            check=True,
        )

    print("14일치 데이터 수집 및 저장 완료")


if __name__ == "__main__":
    main()
