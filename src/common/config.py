import os
from pathlib import Path

try:
    DATA_DIR = Path(os.environ["DATA_DIR"])
except KeyError:
    raise SystemExit("DATA_DIR 환경변수가 없습니다. docs/workflow.md의 '첫 세팅'을 참고하세요.") from None
