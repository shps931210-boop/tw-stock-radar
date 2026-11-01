"""V22 績效優先整改：加權指數開盤價、同時段公平比較、候選實驗的挑股規則與門檻、失準歸因、下一筆訓練標籤。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import daily_learning as dl
from app import fair_compare as fc
from app import miss_deep as md
from app import paper
from app import perf_study as ps
from app import taiex_ohlc as tx


def _D(gap_last=True):
    dates = [f"2026-10-{d:02d}" for d in (14, 15, 16, 17, 20, 21, 22, 27, 28, 29, 30)]
    adj = {}
    for k, sid in enumerate(("0050", "A", "B")):
        c = np.linspace(100, 100 + 4 * (k + 1), len(dates))
        o = c.copy()
        if gap_last:
            o[-1] = c[-2] * 0.97        # 最後一天全市場開盤跳空 −3%
        adj[sid] = pd.DataFrame({"open": o, "close": c, "high": c, "volume": 1e6}, index=dates)
    idx = pd.Series(np.linspace(20000, 20400, len(dates)), index=dates)
    return {"dates": dates, "adj": adj, "idx_s": idx, "info": {}}


class TaiexOhlc(unittest.TestCase):
    def test_parse_roc_slash_dates_and_commas(self):
        js = {"stat": "OK", "fields": ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"],
              "data": [["115/10/30", "46,000.12", "46,100.00", "45,100.00", "45,211.83"], ["115/10/31", "--", "--", "--", "--"]]}
        self.assertEqual(tx.parse(js), [{"date": "2026-10-30", "open": 46000.12, "high": 46100.0, "low": 45100.0, "close": 45211.83}])
        self.assertEqual(tx.parse({"stat": "很抱歉，沒有符合條件的資料!"}), [])

    def test_months_recent_plus_limited_backfill(self):
        from datetime import date
        m = tx.months(date(2026, 11, 1), {"2026-09"})
        self.assertEqual(m[:2], ["2026-11", "2026-10"])
        self.assertNotIn("2026-09", m[2:])
        self.assertLessEqual(len(m), 2 + tx.BACKFILL)


class FairCompare(unittest.TestCase):
    def test_same_window_for_picks_0050_and_taiex(self):
        D = _D()
        d = D["dates"]
        t = {x: (D["idx_s"][x] * 0.99, D["idx_s"][x]) for x in d}
        o = fc.basis(D, ["A", "B"], d[5], d[-1], "open", t, d[5:-1])
        c = fc.basis(D, ["A", "B"], d[4], d[-2], "close", t, d[5:-1])
        a = D["adj"]["0050"]
        self.assertAlmostEqual(o["e0050"], round((a.at[d[-1], "open"] / a.at[d[5], "open"] - 1) * 100, 2))
        self.assertAlmostEqual(c["e0050"], round((a.at[d[-2], "close"] / a.at[d[4], "close"] - 1) * 100, 2))
        self.assertLess(o["e0050"], c["e0050"])          # 開盤跳空只影響開盤口徑
        self.assertIsNone(o["taiex"])                     # 開盤指數不是可成交價格：只列參考，不算相對加權
        self.assertEqual(o["taiex_open_index"], round((t[d[-1]][0] / t[d[5]][0] - 1) * 100, 2))
        self.assertIsNotNone(c["ex_taiex"])
        self.assertAlmostEqual(o["ex_0050"], o["net"] - o["e0050"], places=1)

    def test_missing_taiex_open_is_not_replaced_by_close(self):
        D = _D()
        d = D["dates"]
        o = fc.basis(D, ["A"], d[5], d[-1], "open", {}, d[5:-1])
        self.assertIsNone(o["taiex"])
        self.assertTrue(o["taiex_missing"])
        self.assertIsNone(o["ex_taiex"])


class PerfStudyRules(unittest.TestCase):
    def setUp(self):
        n = 30
        self.X = pd.DataFrame({"rk_aggressive": np.arange(1, n + 1, dtype=float), "sc_aggressive": np.arange(n, 0, -1, dtype=float),
                               "industry": ["PCB"] * 6 + [f"I{k}" for k in range(n - 6)],
                               "vol_1y": np.r_[np.full(5, 90.0), np.full(n - 5, 30.0)], "dist_ma20": np.r_[[50.0], np.zeros(n - 1)],
                               "msurge": np.r_[np.zeros(n - 1), [9.0]], "ret": 0.01, "market_cap": 1e9},
                              index=[f"S{k:02d}" for k in range(n)])
        self.P = {"signal": "2026-10-20", "entry": "2026-10-21", "bench": 0.0}

    def test_industry_cap_vol_cap_and_overheat(self):
        ids, _ = ps.pick({}, self.P, self.X, "aggressive", "R1", False)
        self.assertEqual(sum(self.X.loc[ids, "industry"] == "PCB"), 2)
        ids, _ = ps.pick({}, self.P, self.X, "aggressive", "R2", False)
        self.assertTrue((self.X.loc[ids, "vol_1y"] < 90).all())
        ids, _ = ps.pick({}, self.P, self.X, "aggressive", "C2", False)
        self.assertNotIn("S00", ids)
        ids, _ = ps.pick({}, self.P, self.X, "aggressive", "A", False)
        self.assertEqual(ids, [f"S{k:02d}" for k in range(10)])
        ids, w = ps.pick({}, self.P, self.X, "aggressive", "R4", False)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertLessEqual(w.max(), ps.W_CAP + 1e-9)

    def test_gates_need_both_halves_holdout_and_significance(self):
        k = 40
        dv = np.r_[np.ones(30, bool), np.zeros(10, bool)]
        A = {"rows": [{"ret": 0.0, "big": 0.1} for _ in range(k)], "picks": [[] for _ in range(k)]}
        rng = np.random.default_rng(1)
        good = {"rows": [{"ret": 0.02 + rng.normal(0, 0.002), "big": 0.1} for _ in range(k)], "picks": A["picks"], "_fs": [({"bench": 0}, None)] * k}
        self.assertTrue(ps.compare(good, A, np.zeros(k), dv, "R1")["return_pass"])
        bad = {"rows": [{"ret": (0.02 if i < 15 else -0.01) + rng.normal(0, 0.002), "big": 0.1} for i in range(k)], "picks": A["picks"], "_fs": good["_fs"]}
        self.assertFalse(ps.compare(bad, A, np.zeros(k), dv, "R1")["return_pass"])
        safer = {"rows": [{"ret": -0.001, "big": 0.0 + rng.normal(0, 0.005)} for _ in range(k)], "picks": A["picks"], "_fs": good["_fs"]}
        v = ps.compare(safer, A, np.zeros(k), dv, "R2")
        self.assertTrue(v["risk_pass"])
        self.assertFalse(ps.compare(safer, A, np.zeros(k), dv, "D")["risk_pass"])   # D 是報酬類候選，不能用風控門檻


class MissDeep(unittest.TestCase):
    def test_gap_flag_uses_open_known_at_entry(self):
        D = _D(gap_last=False)
        D["adj"]["A"].loc["2026-10-21", "open"] = D["adj"]["A"].at["2026-10-20", "close"] * 1.05
        s = md.price_state(D, "A", "2026-10-20", "2026-10-21")
        self.assertAlmostEqual(s["gap_entry"], 5.0, places=1)
        self.assertIn("進場跳空≥3%", md.flags({**s, "rev_yoy": 10}))
        self.assertNotIn("最新月營收年減", md.flags({**s, "rev_yoy": 10}))


class LearningLineage(unittest.TestCase):
    def test_next_label_points_to_first_unmatured_research_date(self):
        dates = [f"d{i:03d}" for i in range(100)]
        r = dl.next_label(dates, "d000", h=20)
        self.assertEqual(r["signal_date"], "d080")       # 研究訊號日每 10 個交易日；d080+21 > 99
        self.assertEqual(r["trading_days_left"], 80 + 21 - 99)

    def test_attribution_written_once(self):
        D = _D(gap_last=False)
        tmp = Path(tempfile.mkdtemp())
        mat, out, picks = tmp / "m", tmp / "a", tmp / "p"
        for p in (mat, picks):
            p.mkdir()
        (mat / "2026-10-20_h5.json").write_text(json.dumps({"signal_date": "2026-10-20", "horizon": 5, "entry": "2026-10-21", "exit": "2026-10-30",
                                                            "official": {"file": "2026-10-20.json"}}), "utf-8")
        (picks / "2026-10-20.json").write_text(json.dumps({"picks": {m: [{"stock_id": "A", "rank": 1}, {"stock_id": "B", "rank": 2}] for m in md.MODES}}), "utf-8")
        o = lambda sid, a, b: D["adj"][sid].at[b, "open"] / D["adj"][sid].at[a, "open"] - 1
        old = md.PICKS, md.pb.hold_return, md.pb._bench
        md.PICKS, md.pb.hold_return, md.pb._bench = picks, (lambda D_, sid, a, b: (o(sid, a, b), "ok")), (lambda D_, a, b: o("0050", a, b))
        try:
            r1 = dl.attribute(D, mat, out)
            r2 = dl.attribute(D, mat, out)
        finally:
            md.PICKS, md.pb.hold_return, md.pb._bench = old
        self.assertEqual(len(r1["new"]), 1)
        self.assertEqual(r2["new"], [])
        self.assertEqual(r2["total"], 1)


class PaperBenchStart(unittest.TestCase):
    def test_taiex_uses_same_open_start_when_available(self):
        days = ["2026-10-27", "2026-10-28", "2026-10-29"]
        px = {"0050": {d: (100.0, 100.0, 100.0, 100.0, 1e6) for d in days}}
        idx = {"2026-10-27": 100.0, "2026-10-28": 102.0, "2026-10-29": 101.0}
        b = paper._bench_curve(days, px, "2026-10-28", {}, idx, {"2026-10-28": 101.0})
        self.assertEqual(b["taiex_start"], "open")
        self.assertAlmostEqual(b["taiex"]["ret"], 0.0, places=2)
        b = paper._bench_curve(days, px, "2026-10-28", {}, idx, {})
        self.assertEqual(b["taiex_start"], "prev_close")
        self.assertAlmostEqual(b["taiex"]["ret"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
