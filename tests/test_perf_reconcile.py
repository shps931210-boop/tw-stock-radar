"""V21.2 正式績效對帳：獨立重算要和到期對帳檔一致；被改過的數字、名單、進出場日要被抓出來。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import perf_reconcile as pr


def _D():
    dates = [f"2026-10-{d:02d}" for d in (14, 15, 16, 17, 20, 21, 22, 24, 27, 28, 29)]   # 10/23 颱風休市、10/26 補假不在交易日曆
    adj = {}
    for k, sid in enumerate(("0050", "A", "B", "C")):
        px = np.linspace(100, 100 + 3 * (k - 1), len(dates))
        px[-1] = px[-2]   # 最後一天報酬 0：零報酬不能被當成缺值
        adj[sid] = pd.DataFrame({"open": px, "close": px}, index=dates)
    return {"dates": dates, "adj": adj}


def _rec(D, sig="2026-10-16", h=5):
    d = D["dates"]
    i = d.index(sig)
    e, x = d[i + 1], d[i + 1 + h]
    o = lambda s, t: D["adj"][s].loc[t, "open"]
    b = o("0050", x) / o("0050", e) - 1
    rows = []
    for s in ("A", "B", "C"):
        r = o(s, x) / o(s, e) - 1
        rows.append({"id": s, "status": "ok", "ret": round(r * 100, 2), "net": round((r - pr.COST) * 100, 2),
                     "excess": round((r - pr.COST - b) * 100, 2)})
    return {"signal_date": sig, "horizon": h, "entry": e, "exit": x, "bench_0050": round(b * 100, 2), "cost_pct": 0.585,
            "official": {"file": f"{sig}.json", "modes": {"stable": {"stocks": rows, "mean_excess": round(float(np.mean([y["excess"] for y in rows])), 2)}}}}


class Reconcile(unittest.TestCase):
    def setUp(self):
        self.D = _D()
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "2026-10-16.json").write_text(json.dumps({"picks": {"stable": [{"stock_id": s} for s in "ABC"]}}), "utf-8")

    def test_consistent_passes_and_holiday_window(self):
        r = _rec(self.D)
        self.assertEqual((r["entry"], r["exit"]), ("2026-10-17", "2026-10-27"))   # 跳過颱風假與補假
        self.assertEqual(pr.check_file(self.D, r, self.tmp), [])

    def test_tampered_numbers_list_and_window_are_caught(self):
        base = _rec(self.D)
        r = copy.deepcopy(base); r["official"]["modes"]["stable"]["mean_excess"] += 0.5
        self.assertTrue(any("平均超額" in x for x in pr.check_file(self.D, r, self.tmp)))
        r = copy.deepcopy(base); r["official"]["modes"]["stable"]["stocks"][0]["ret"] += 1
        self.assertTrue(pr.check_file(self.D, r, self.tmp))
        r = copy.deepcopy(base); r["official"]["modes"]["stable"]["stocks"].reverse()
        self.assertTrue(any("名單" in x for x in pr.check_file(self.D, r, self.tmp)))
        r = copy.deepcopy(base); r["exit"] = "2026-10-28"
        self.assertTrue(any("進出場日" in x for x in pr.check_file(self.D, r, self.tmp)))
        r = copy.deepcopy(base); r["cost_pct"] = 0.3
        self.assertTrue(any("成本" in x for x in pr.check_file(self.D, r, self.tmp)))

    def test_definitions_separate_every_kind(self):
        self.assertTrue({"official", "shadow", "backtest", "paper", "factor", "taiex"} <= set(pr.DEFINITIONS))


class Eligibility(unittest.TestCase):
    """V21.2（10/29 驗收 P0）：開盤後才產生、沒有產生時間、資料日不對的名單，不准進任何績效統計。"""
    def test_eligible_rules(self):
        from app import track
        ok = {"signal_date": "2026-10-16", "generated_at": "2026-10-16T14:52:57", "data_dates": {"prices": "2026-10-16"}}
        self.assertEqual(track.perf_eligible(ok, "2026-10-19"), (True, None))
        self.assertFalse(track.perf_eligible({**ok, "generated_at": "2026-10-19T09:00:01"}, "2026-10-19")[0])
        self.assertFalse(track.perf_eligible({**ok, "generated_at": None}, "2026-10-19")[0])
        self.assertFalse(track.perf_eligible({**ok, "data_dates": {"prices": "2026-10-15"}}, "2026-10-19")[0])

    def test_gate_clears_stats_and_violations_zero(self):
        from app import maturity_report as mr
        r = {"official": {"on_time": False, "modes": {"stable": {"mean_excess": -3.6, "mean_net": -2.8, "stocks": [1]}}},
             "shadow": {"eligible": True, "models": {"v21": {"mean_excess": 0.47, "ic": 0.08}}}}
        g = mr.gate(r)
        self.assertIsNone(g["official"]["modes"]["stable"]["mean_excess"])
        self.assertEqual(g["official"]["modes"]["stable"]["stocks"], [1])          # 明細保留供查核
        self.assertEqual(r["official"]["modes"]["stable"]["mean_excess"], -3.6)    # 原檔不改
        self.assertEqual(g["shadow"]["models"]["v21"]["mean_excess"], 0.47)
        idx = [{"eligible": {"official": False}, "official": {"stable": {"mean_excess": -3.6}}}]
        self.assertEqual(mr.violations(idx), 1)
        idx = [{"eligible": {"official": False}, "official": g["official"]["modes"]}]
        self.assertEqual(mr.violations(idx), 0)
