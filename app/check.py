"""檢查資料是否最新：python -m app.check（GitHub Actions 用，過期或有來源失敗就回傳錯誤碼，GitHub 會寄失敗通知）。"""
import json
import sys

from . import config


def main() -> int:
    p = config.DATA_DIR / "snapshot.json"
    if not p.exists():
        print("❌ 沒有真實資料快照")
        return 1
    s = json.loads(p.read_text("utf-8"))
    problems = []
    price = s["data_dates"]["prices"]
    if s.get("expected_date") and price and price < s["expected_date"]:
        basis = "證交所已公布" if s.get("expected_basis") == "official" else "推估應有"
        problems.append(f"股價只到 {price}，{basis} {s['expected_date']}")
    for h in s.get("health", []):
        if h["source"] in ("twse_quotes", "tpex_quotes") and (not h["last_ok"] or h["last_ok"] < (price or "")):
            problems.append(f"{h['label']}最後成功：{h['last_ok'] or '從未成功'}")
    if not s.get("history", {}).get("ready", True):
        print(f"⏳ 歷史股價回補中：{s['history']['days']}/{s['history']['need']} 天")
    for x in problems:
        print("❌", x)
    if not problems:
        print(f"✅ 資料最新：{price}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
