"""檢查資料是否最新、備份是否成功：python -m app.check（GitHub Actions 用，過期或有來源失敗就回傳錯誤碼，GitHub 會寄失敗通知）。"""
import json
import os
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
    # 夜間財報補齊不抓行情：行情是否最新由盤後更新負責（那邊失敗會寄信），這裡只提醒，避免整晚重複寄信
    backfill = os.environ.get("MODE") == "backfill"
    if s.get("expected_date") and price and price < s["expected_date"]:
        basis = "證交所已公布" if s.get("expected_basis") == "official" else "推估應有"
        (print if backfill else problems.append)(f"股價只到 {price}，{basis} {s['expected_date']}")
    # 14:10、14:40 的快速更新時，櫃買中心常常還沒公布：16:00 前只提醒、不算失敗（不寄失敗通知信）
    early = os.environ.get("MODE") == "fast" and config.now().hour < 16 or backfill
    for h in s.get("health", []):
        if h["source"] in ("twse_quotes", "tpex_quotes") and (not h["last_ok"] or h["last_ok"] < (price or "")):
            msg = f"{h['label']}最後成功：{h['last_ok'] or '從未成功'}"
            if early:
                print(f"⏳ {msg}（16:00 前的快速更新，下一次排程會再補）")
            else:
                problems.append(msg)
    b = config.DATA_DIR / "backup_status.json"
    if b.exists():
        bs = json.loads(b.read_text("utf-8"))
        if not bs.get("ok"):
            problems.append("資料庫備份或還原測試失敗：" + "；".join(bs.get("problems") or ["原因不明"]) + f"（最後一次成功：{bs.get('last_ok') or '從未'}）")
    if os.environ.get("MODE") == "full":  # V31：每日自主學習連續失敗就寄通知信（網站照樣發布）
        from . import daily_learning
        st = daily_learning.run_status()
        if st.get("alert"):
            problems.append(f"每日自主學習連續失敗 {st['fail_streak']} 次：{(st.get('last') or {}).get('error') or '原因不明'}（最近一次成功：{st.get('last_ok') or '從未'}）")
        if st.get("partial_alert"):
            steps = "、".join((st.get("last") or {}).get("failed_steps") or [])
            problems.append(f"每日自主學習連續 {st['partial_streak']} 次有步驟失敗（{steps}）；最後一次完整成功：{st.get('last_valid_report') or '從未'}")
    if not s.get("history", {}).get("ready", True):
        print(f"⏳ 歷史股價回補中：{s['history']['days']}/{s['history']['need']} 天")
    for x in problems:
        print("❌", x)
    if not problems:
        print(f"✅ 資料最新：{price}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
