import os
from pathlib import Path


def _required(name):
    try:
        return os.environ[name]
    except KeyError:
        raise SystemExit(f"{name} 환경변수가 없습니다. docs/workflow.md의 '첫 세팅'을 참고하세요.") from None


DATA_DIR = Path(_required("DATA_DIR"))
MACHINE = _required("MACHINE")  # kyle-win, eunbin-mac, colab-cpu, colab-T4 — 결과 CSV의 machine 컬럼
