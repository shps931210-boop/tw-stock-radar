"""智慧警報：用 Node 執行網頁裡的 alertEvents 純函式（static/index.html 的 <alert-core> 區塊）。"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

HTML = (Path(__file__).parent.parent / "static" / "index.html").read_text("utf-8")
CORE = re.search(r"const ALERT_TYPES = \{.*?^\};", HTML, re.S | re.M).group(0) + "\n" + \
    re.search(r"// <alert-core>(.*?)// </alert-core>", HTML, re.S).group(1)


def stock(sid, score, rank, status="none", ok=True, ex=(), date="2026-10-12"):
    return {"stock_id": sid, "name": f"股{sid}", "date": date, "base_ok": ok, "exclude_reasons": list(ex),
            "ai": {"modes": {"balanced": {"score": score, "rank": rank}}}, "timing": {"status": status, "triggers": ["sig_breakout"]}}


def run(snap, watch, rules, state, today="2026-10-12"):
    js = CORE + f"\nconsole.log(JSON.stringify(alertEvents({json.dumps(snap)}, {json.dumps(watch)}, {json.dumps(rules)}, {json.dumps(state)}, {json.dumps(today)})));"
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@unittest.skipUnless(shutil.which("node"), "需要 Node")
class Alerts(unittest.TestCase):
    def snap(self, stocks, events=()):
        return {"data_dates": {"prices": "2026-10-12"}, "stocks": stocks, "triggers": {"sig_breakout": "帶量突破"},
                "calendar": {"events": list(events)}}

    def test_first_run_only_records_baseline(self):
        r = run(self.snap([stock("2330", 70, 5)]), ["2330"], {}, None)
        self.assertEqual(r["notes"], [])
        self.assertEqual(r["state"], {"date": "2026-10-12", "base": {"2330": {"s": 70, "r": 5, "ok": True, "ex": []}}})

    def test_changes_since_last_date(self):
        prev = {"date": "2026-10-11", "base": {"2330": {"s": 55, "r": 40, "ok": True, "ex": []}, "2317": {"s": 60, "r": 30, "ok": True, "ex": []}}}
        snap = self.snap([stock("2330", 70, 5, "buy"), stock("2317", 58, 31, ok=False, ex=["負債比過高"]), stock("1101", 90, 1)])
        keys = sorted(n["key"] for n in run(snap, ["2330", "2317"], {}, prev)["notes"])
        self.assertEqual(keys, ["risk:2317:2026-10-12", "score:2330:2026-10-12", "signal:2330:2026-10-12", "top:2330:2026-10-12"])

    def test_same_day_rerun_is_idempotent(self):
        prev = {"date": "2026-10-11", "base": {"2330": {"s": 55, "r": 40, "ok": True, "ex": []}}}
        snap = self.snap([stock("2330", 70, 5)])
        a = run(snap, ["2330"], {}, prev)
        b = run(snap, ["2330"], {}, a["state"])   # 同一個資料日再跑一次：不產生新的比較通知
        self.assertTrue(a["notes"])
        self.assertEqual(b["notes"], [])
        self.assertEqual(b["state"], a["state"])

    def test_rules_disable_and_threshold(self):
        prev = {"date": "2026-10-11", "base": {"2330": {"s": 65, "r": 40, "ok": True, "ex": []}}}
        snap = self.snap([stock("2330", 70, 30, "buy")])
        self.assertEqual(run(snap, ["2330"], {"signal": {"enabled": False}}, prev)["notes"], [])   # 5 分 < 預設 10 分
        keys = [n["key"] for n in run(snap, ["2330"], {"signal": {"enabled": False}, "score": {"enabled": True, "params": {"min": 5}}}, prev)["notes"]]
        self.assertEqual(keys, ["score:2330:2026-10-12"])

    def test_exdiv_only_confirmed_within_days(self):
        ev = [{"event_type": "dividend", "status": "confirmed", "stock_id": "2330", "event_date": "2026-10-14", "title": "除息（現金股利 5 元）"},
              {"event_type": "dividend", "status": "unverified", "stock_id": "2330", "event_date": "2026-10-13", "title": "除息"},
              {"event_type": "dividend", "status": "confirmed", "stock_id": "2330", "event_date": "2026-10-30", "title": "除息"},
              {"event_type": "dividend", "status": "confirmed", "stock_id": "9999", "event_date": "2026-10-13", "title": "除息"}]
        keys = [n["key"] for n in run(self.snap([stock("2330", 70, 5)], ev), ["2330"], {}, None)["notes"]]
        self.assertEqual(keys, ["exdiv:2330:2026-10-14"])

    def test_stale_price_no_signal(self):
        snap = self.snap([stock("2330", 70, 5, "buy", date="2026-10-08")])
        self.assertEqual(run(snap, ["2330"], {}, None)["notes"], [])


if __name__ == "__main__":
    unittest.main()
