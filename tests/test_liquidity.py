"""V27 流動性風險控制：高分冷門股不會成為最高推薦、仍可搜尋、缺資料不當成流動、回測與網站同一套規則、原始分數不被改。"""
import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import learning, liquidity, metrics
from app import portfolio_research as pr


def stock(sid, score, avg, med=None, rank=None, vol=50, low=0, zero=0, days=20, ind="電子"):
    return {"stock_id": sid, "industry": ind, "avg_value_20": avg, "value_med_20": avg if med is None else med,
            "low_value_days_20": low, "zero_vol_days_60": zero, "value_days_20": days, "vol_1y_pct": vol, "is_common": True,
            "ai": {"modes": {"balanced": {"score": score, "rank": rank}}, "risks": [], "strengths": []}}


def universe():
    # 第一名分數最高但成交金額只有 1,500 萬（E 級）；第二名是 D 級；其餘是流動性 A 的股票
    rows = [stock("9001", 93, 1.5e7), stock("9002", 92, 5e7)]
    rows += [stock(f"1{i:03d}", 90 - i, 2e9, ind=f"產業{i % 5}") for i in range(15)]
    for i, r in enumerate(sorted(rows, key=lambda r: -r["ai"]["modes"]["balanced"]["score"]), 1):
        r["ai"]["modes"]["balanced"]["rank"] = i
    return rows


class Grades(unittest.TestCase):
    def test_thresholds(self):
        for v, g in ((1e9, "A"), (3e8, "B"), (1e8, "C"), (3e7, "D"), (1e7, "E"), (9e6, "F")):
            self.assertEqual(liquidity.grade(stock("x", 50, v))["grade"], g)

    def test_missing_data_is_not_liquid(self):
        self.assertEqual(liquidity.grade({"avg_value_20": None})["grade"], "?")
        self.assertEqual(liquidity.grade(stock("x", 50, 5e9, days=3))["grade"], "?")
        st, why = liquidity.status("?", 10)
        self.assertEqual(st, "excluded")
        self.assertIn("資料不足", why)

    def test_one_day_spike_uses_median(self):
        g = liquidity.grade(stock("x", 50, 4e7, med=4e6))   # 平均 4,000 萬，但大部分日子只有 400 萬
        self.assertEqual(g["base_grade"], "D")
        self.assertEqual(g["grade"], "F")

    def test_many_low_days_downgrade(self):
        self.assertEqual(liquidity.grade(stock("x", 50, 2e8, low=6))["grade"], "D")
        self.assertEqual(liquidity.grade(stock("x", 50, 2e8, zero=3))["grade"], "D")


class Recommend(unittest.TestCase):
    def setUp(self):
        self.rows = universe()
        self.before = copy.deepcopy([r["ai"]["modes"] for r in self.rows])
        liquidity.annotate(self.rows, ["balanced"])
        self.by = {r["stock_id"]: r for r in self.rows}

    def test_high_score_illiquid_is_never_top_recommendation(self):
        rec = sorted((r for r in self.rows if r["ai"]["rec"]["balanced"]["rank"]), key=lambda r: r["ai"]["rec"]["balanced"]["rank"])
        top = [r["stock_id"] for r in rec[:liquidity.TOP]]
        self.assertNotIn("9001", top)   # E 級：完全不列入
        self.assertNotIn("9002", top)   # D 級：不進前 10 名
        self.assertEqual(self.by["9001"]["ai"]["rec"]["balanced"]["recommendation_eligibility"], "excluded")
        d = self.by["9002"]["ai"]["rec"]["balanced"]
        self.assertEqual(d["recommendation_eligibility"], "capped")
        self.assertEqual(d["rank"], liquidity.TOP + 1)
        self.assertLessEqual(d["recommendation_score"], rec[liquidity.TOP - 1]["ai"]["modes"]["balanced"]["score"])

    def test_illiquid_stock_is_still_searchable_with_raw_score(self):
        r = self.by["9001"]
        self.assertEqual(r["ai"]["modes"]["balanced"], {"score": 93, "rank": 1})   # 原始分數與名次照舊
        self.assertEqual(r["liquidity_grade"], "E")
        self.assertIsNotNone(r["ai"]["rec"]["balanced"]["recommendation_cap_reason"])

    def test_raw_scores_not_polluted(self):
        self.assertEqual([r["ai"]["modes"] for r in self.rows], self.before)

    def test_backtest_uses_same_rule(self):
        live = [r["stock_id"] for r in sorted((r for r in self.rows if r["ai"]["rec"]["balanced"]["rank"]),
                                              key=lambda r: r["ai"]["rec"]["balanced"]["rank"])][:12]
        rows = universe()
        for r in rows:
            r["liq"] = liquidity.grade(r)
        self.assertEqual(pr.pick(rows, "balanced", liquidity.LIVE, 12), live)
        self.assertEqual(pr.pick(rows, "balanced", "A", 3), ["9001", "9002", "1000"])

    def test_variant_d_excludes_d_with_high_volatility(self):
        self.assertEqual(liquidity.status("D", 85, "D")[0], "excluded")
        self.assertEqual(liquidity.status("D", 85, "C")[0], "capped")
        self.assertEqual(liquidity.status("C", 95, "D")[0], "capped")
        self.assertEqual(liquidity.status("E", 10, "A")[0], "normal")

    def test_alerts(self):
        r = stock("x", 90, 2e9)
        r.update(vol_ratio=4, ret_20d=35, fin_stale=True)
        r["liq"] = liquidity.grade(r)
        kinds = {a["kind"] for a in liquidity.alerts(r)}
        self.assertEqual(kinds, {"volume_spike", "run_up", "stale"})


class Metrics(unittest.TestCase):
    def test_price_metrics_liquidity_fields(self):
        n = 70
        val = np.full(n, 2e7)
        val[-20:-5] = 1e6
        vol = np.full(n, 1000.0)
        vol[-3:] = 0
        px = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d"), "close": 10.0, "high": 10.0,
                           "low": 10.0, "volume": vol, "value": val, "open": 10.0})
        m = metrics.price_metrics(px, None)
        self.assertEqual(m["low_value_days_20"], 15)
        self.assertEqual(m["zero_vol_days_60"], 3)
        self.assertEqual(m["value_med_20"], 1e6)
        self.assertEqual(liquidity.grade(m)["grade"], "F")


class Acceptance(unittest.TestCase):
    def test_technical_pass_alone_cannot_promote(self):
        act = {"version": "v1", "problems": []}
        g = {"challenger": "lgbm", "status": "passed", "label": "通過"}
        res = {"horizons": {"20": {"vs_v21": {"lgbm": {"diff": 0.01, "t": 0.5}}, "holdout": {"lgbm": {"ic": 0.08}, "v21": {"ic": 0.03}}}}}
        a = learning.acceptance(act, res, g)
        self.assertEqual([x["ok"] for x in a["stages"]], [True, False, True])
        self.assertFalse(a["ok"])
        res["horizons"]["20"]["vs_v21"]["lgbm"]["t"] = 3.0
        self.assertTrue(learning.acceptance(act, res, g)["ok"])
        self.assertFalse(learning.acceptance(act, res, {**g, "status": "kept"})["ok"])


class SealCompleteness(unittest.TestCase):
    def test_incomplete_market_is_not_sealed(self):
        from app import ml_predict
        con = sqlite3.connect(":memory:")
        con.executescript("CREATE TABLE stocks (stock_id TEXT, market TEXT); CREATE TABLE prices (stock_id TEXT, date TEXT);")
        con.executemany("INSERT INTO stocks VALUES (?,?)", [(f"a{i}", "twse") for i in range(100)] + [(f"b{i}", "tpex") for i in range(100)])
        con.executemany("INSERT INTO prices VALUES (?,?)", [(f"{m}{i}", "2026-10-15") for m in "ab" for i in range(100)])
        con.executemany("INSERT INTO prices VALUES (?,?)", [(f"a{i}", "2026-10-16") for i in range(100)])  # 櫃買還沒公布
        c = ml_predict.completeness(con, "2026-10-16")
        self.assertFalse(c["ok"])
        self.assertIn("上櫃", c["why"])
        con.executemany("INSERT INTO prices VALUES (?,?)", [(f"b{i}", "2026-10-16") for i in range(97)])
        self.assertTrue(ml_predict.completeness(con, "2026-10-16")["ok"])


class PremarketQuality(unittest.TestCase):
    def test_duplicates_counted_once_and_quality_flags_failures(self):
        from datetime import datetime
        from app import premarket as pm
        ev = [{"id": "a", "title": "本公司董事會決議停工", "affected_tickers": ["2330"], "first_seen_at": "2026-10-17T10:00", "published_at": "2026-10-17T09:00"},
              {"id": "b", "title": "本公司董事會決議停工", "affected_tickers": ["2330"], "first_seen_at": "2026-10-18T10:00", "published_at": "2026-10-18T09:00"},
              {"id": "c", "title": "本公司董事會決議停工", "affected_tickers": ["2603"], "first_seen_at": "2026-10-18T10:00", "published_at": "2026-10-18T09:00"}]
        self.assertEqual(pm.dedupe(ev), 1)
        self.assertEqual(ev[1]["duplicate_of"], "a")
        q = pm.check_quality(datetime(2026, 10, 19, 8, 15), {}, {"ok": True}, None, {"stale": []}, 1, ev)
        self.assertEqual(q["state"], "degraded")
        self.assertFalse(q["sources"]["global"]["ok"])


if __name__ == "__main__":
    unittest.main()
