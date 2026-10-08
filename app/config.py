import json
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# 所有日期時間都用台灣時間，不依賴主機的系統時區（國外主機、GitHub Actions 通常是 UTC）。
# 程式一律呼叫 config.now() / config.today()，不要直接用 datetime.now() / date.today()。
TAIPEI = ZoneInfo("Asia/Taipei")


def now() -> datetime:
    """台灣目前時間（不帶時區資訊，方便和資料庫裡的時間字串比較）。"""
    return datetime.now(TAIPEI).replace(tzinfo=None)


def today() -> date:
    return now().date()


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load() -> dict:
    cfg = json.loads((ROOT / "config.json").read_text("utf-8"))
    if os.environ.get("FINMIND_TOKEN"):  # 部署時用環境變數，不要把 token 寫進程式庫
        cfg["finmind_token"] = os.environ["FINMIND_TOKEN"]
    return cfg
