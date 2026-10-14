"""V19.1 個股研究分級顯示：單一欄位缺漏不讓整張研究卡失效，但排行榜的完整性要求不能被放寬。"""
import unittest
from datetime import date

import numpy as np
import pandas as pd

from app import config, coverage, snapshot, strategy, valuation


def frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({"stock_id": f"{1000 + i}", "name": f"S{i}", "industry": "電子" if i % 2 else "塑膠", "market": "twse",
                     "is_common": True, "is_financial": False, "disposition": False, "long_term_loss": False,
                     "avg_value_20": 1e9, "history_days": 300, "date": "2026-10-12", "close": 50.0,
                     "roe": rng.normal(10, 5), "roa": rng.normal(5, 2), "op_margin": rng.normal(10, 5), "ocf_to_assets": rng.normal(5, 2),
                     "debt_ratio": rng.normal(40, 10), "rev_yoy_3m": rng.normal(5, 10), "rev_yoy": rng.normal(5, 10),
                     "eps_q_yoy": rng.normal(5, 20), "op_q_yoy": rng.normal(5, 20), "earnings_yield": rng.normal(5, 2),
                     "book_to_price": rng.normal(0.6, 0.2), "fcf_yield": rng.normal(3, 2), "mom_12_1": rng.normal(10, 20),
                     "ma20": 48, "ma60": 46, "ma120": 44, "ma240": 42, "ma60_slope_20": 1.0,
                     "fin_period_end": "2026-06-30", "rev_ym": "2026-09", "val_date": "2026-10-12"})
    return pd.DataFrame(rows)


class RankingStaysStrict(unittest.TestCase):
    def setUp(self):
        df = frame()
        df.loc[0, "avg_value_20"] = 1e5          # 成交金額過低：不進排行榜
        df.loc[1, ["roe", "roa"]] = np.nan       # 通過門檻但缺品質必要欄位
        df.loc[2, ["roe", "roa", "rev_yoy", "rev_yoy_3m"]] = np.nan
        df.loc[2, "avg_value_20"] = 1e5
        self.df, _ = snapshot.score_frame(df, config.load(), date(2026, 10, 14))

    def test_unranked_gets_reference_scores_but_no_composite(self):
        r = self.df.iloc[0]
        self.assertFalse(r["base_ok"])
        self.assertTrue(pd.isna(r["composite"]))
        self.assertEqual(set(r["ref_scores"]), {"quality", "growth", "value", "momentum", "trend"})
        self.assertTrue(all(0 <= v <= 100 for v in r["ref_scores"].values()))

    def test_reference_scores_follow_required_fields(self):
        r = self.df.iloc[2]
        self.assertNotIn("quality", r["ref_scores"])  # 缺 ROE、ROA 不給品質分數
        self.assertNotIn("growth", r["ref_scores"])

    def test_missing_factor_still_blocks_full_ranking(self):
        r = self.df.iloc[1]
        self.assertTrue(r["base_ok"])
        self.assertFalse(r["complete"])
        self.assertTrue(pd.isna(r["composite"]))
        self.assertIsNone(r["ref_scores"])  # 進排行榜流程的股票用正式分數，不另外算參考分數

    def test_ranked_stocks_have_no_reference_scores(self):
        self.assertTrue(all(x is None for x in self.df[self.df["base_ok"]]["ref_scores"]))


class ValuationLevels(unittest.TestCase):
    def test_statuses(self):
        fin = pd.DataFrame({"period": [f"{y}-Q{q}" for y in (2023, 2024) for q in (1, 2, 3, 4)], "eps": [1] * 8})
        d = pd.bdate_range("2025-04-01", periods=60).strftime("%Y-%m-%d")
        raw = pd.DataFrame({"date": d, "close": np.linspace(40, 60, 60)})
        b = valuation.band(raw, fin, False)
        self.assertEqual(b["status"], "insufficient_history")
        self.assertIsNotNone(b["pe"])        # 歷史不夠仍有目前本益比
        self.assertNotIn("p50", b)            # 但不假裝有歷史區間
        self.assertEqual(valuation.band(raw, fin.iloc[:2], False)["status"], "missing_source")
        loss = fin.assign(eps=-1)
        self.assertEqual(valuation.band(raw, loss, False)["status"], "not_applicable")

    def test_official_pe_fallback(self):
        v = snapshot._valuation_fill({"per": 12.5, "pbr": 1.3, "dy": 4.0, "val_date": "2026-10-12"},
                                     {"status": "missing_source", "why": "沒有連續四季 EPS"})
        self.assertEqual(v["status"], "partial")
        self.assertEqual((v["pe"], v["pe_src"]), (12.5, "official"))
        self.assertNotIn("p50", v)
        # 虧損公司不補本益比
        v = snapshot._valuation_fill({"per": 0, "pbr": 0.9}, {"status": "not_applicable", "why": "虧損"})
        self.assertNotIn("pe", v)
        self.assertEqual(v["pbr"], 0.9)


class LongPartial(unittest.TestCase):
    base = {"base_ok": True, "complete": True, "score_quality": 70, "score_growth": 75, "close": 100, "ma240": 90,
            "ai": {"modes": {"balanced": {"score": 70}}}}

    def test_full(self):
        lg = strategy.long_status(self.base, {"pe": 10, "p50": 12, "p80": 15})
        self.assertEqual((lg["state"], lg["status"]), ("triggered", "ready"))

    def test_missing_growth_is_partial_and_capped_at_setup(self):
        s = {**self.base, "score_growth": None, "complete": False, "missing_detail": ["成長能力（缺EPS 年增）"]}
        lg = strategy.long_status(s, {"pe": 10, "p50": 12})
        self.assertEqual(lg["status"], "partial")
        self.assertEqual(lg["state"], "setup")
        self.assertIn("成長能力（缺EPS 年增）", lg["miss"])

    def test_failed_item_still_fails(self):
        lg = strategy.long_status({**self.base, "score_quality": 30, "score_growth": None, "complete": False}, None)
        self.assertEqual(lg["state"], "none")

    def test_unranked_uses_reference_scores(self):
        s = {"base_ok": False, "exclude_reasons": ["成交金額過低"], "ref_scores": {"quality": 80, "growth": 70},
             "close": 100, "ma240": 90, "ai": {"modes": {"balanced": {"score": None}}}}
        lg = strategy.long_status(s, {"pe": 10, "peer": 15})
        self.assertEqual(lg["status"], "partial")
        self.assertEqual(lg["state"], "setup")
        self.assertEqual(lg["basis"], "ref")

    def test_nothing_available(self):
        lg = strategy.long_status({"base_ok": False, "ref_scores": {"value": 50}}, None)
        self.assertEqual((lg["state"], lg["status"]), ("na", "missing_source"))


class Modules(unittest.TestCase):
    def test_each_module_independent(self):
        r = {"stock_id": "1103", "date": "2026-10-12", "fin_period": None, "rev_ym": "2026-09", "base_ok": False,
             "exclude_reasons": ["成交金額過低"], "ref_scores": {"value": 90}}
        st = {"ready": True, "status": "ready", "short": {"state": "none"}, "swing": {"state": "none"}}
        M = snapshot._modules(r, st, {"status": "partial", "why": "沒有連續四季 EPS"}, {"state": "na", "status": "missing_source", "miss": ["缺財報"]},
                              "2026-10-12", set())
        self.assertEqual(M["quote"]["status"], "ready")
        self.assertEqual(M["fundamental"]["status"], "partial")
        self.assertIn("補齊佇列", M["fundamental"]["why"])
        self.assertEqual(M["short"]["status"], "ready")
        self.assertEqual(M["ai"]["status"], "partial")
        self.assertTrue(all(m["label"] for m in M.values()))
        self.assertEqual(snapshot._modules({**r, "date": "2026-09-01"}, st, {"status": "ready"}, {"state": "none"}, "2026-10-12", None)["quote"]["status"], "stale")

    def test_coverage_report(self):
        rows = [{"stock_id": "1", "is_common": True, "market_cap": 1e12, "market": "twse", "base_ok": True, "history_days": 300},
                {"stock_id": "2", "is_common": True, "market_cap": 1e9, "market": "tpex", "base_ok": False, "history_days": 30}]
        dec = {"1": {"modules": {"quote": {"status": "ready"}, "valuation": {"status": "ready"}}},
               "2": {"modules": {"quote": {"status": "ready"}, "valuation": {"status": "missing_source", "why": "可用的歷史本益比只有 30 天"}}}}
        c = coverage.build(rows, dec)
        self.assertEqual(c["stocks"], 2)
        self.assertEqual(c["modules"]["valuation"]["counts"], {"ready": 1, "missing_source": 1})
        self.assertEqual(c["reasons"][0]["why"], "可用的歷史本益比只有 N 天")
        self.assertEqual(c["samples"]["大型權值股"]["stock_id"], "1")
        self.assertEqual(c["samples"]["新上市股票"]["stock_id"], "2")


if __name__ == "__main__":
    unittest.main()
