"""V20.4：到期對帳明細只寫一次、財報重新評分的原因、母體外綜合分數不影響正式排名。"""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app import fin_watch, maturity_report

DATES = ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-27", "2026-10-28"]


def _D(prices: dict) -> dict:
    adj, raw = {}, {}
    for sid, opens in prices.items():
        df = pd.DataFrame({"date": DATES, "open": opens, "high": opens, "low": opens, "close": opens,
                           "volume": [1e6] * len(DATES), "value": [1e9] * len(DATES)}).set_index("date")
        # 開高低收不全相同，避免被當成漲停鎖死
        df["high"] = df["open"] * 1.01
        adj[sid] = raw[sid] = df
    return {"dates": DATES, "adj": adj, "rawi": raw, "idx_s": pd.Series([100.0] * len(DATES), index=DATES),
            "info": {k: {"name": "股" + k} for k in prices}}


class Maturity(unittest.TestCase):
    def test_build_once_and_metrics(self):
        up = [100, 100, 100, 101, 102, 103, 104, 110]
        D = _D({"0050": [100] * 8, **{f"{i:04d}": up for i in range(1, 41)}, "9999": [100] * 8})
        with tempfile.TemporaryDirectory() as t:
            T = Path(t)
            pred, picks, out = T / "p", T / "k", T / "o"
            pred.mkdir(); picks.mkdir()
            stocks = [{"id": f"{i:04d}", "5": {"v21": i, "ridge": i, "lgbm": i}} for i in range(1, 41)]
            (pred / "2026-10-16.json").write_text(json.dumps({"signal_date": "2026-10-16", "generated_at": "2026-10-16T15:00:00", "n": 40, "stocks": stocks}))
            (picks / "2026-10-16.json").write_text(json.dumps({"generated_at": "2026-10-16T15:00:00",
                                                                "rec_picks": {"balanced": [{"stock_id": "0001", "rank": 1}, {"stock_id": "9999", "rank": 2}]}}))
            I = maturity_report.run(pred=pred, picks=picks, out=out, D=D)
            self.assertEqual(I["new"], ["2026-10-16_h5.json"])
            R = json.loads((out / "2026-10-16_h5.json").read_text())
            self.assertEqual((R["entry"], R["exit"]), ("2026-10-19", "2026-10-28"))   # 隔天開盤進、第 5 個交易日後開盤出
            b = R["official"]["modes"]["balanced"]
            self.assertEqual(b["filled"], 2)
            self.assertAlmostEqual(b["stocks"][0]["net"], 10 - 0.585, delta=0.01)
            self.assertEqual(b["up_hit"], 50.0)
            self.assertTrue(R["official"]["on_time"])
            self.assertEqual(R["shadow"]["models"]["v21"]["n"], 10)
            # 第二次執行不改已寫的批次
            (out / "2026-10-16_h5.json").write_text('{"signal_date":"2026-10-16","horizon":5,"exit":"x","bench_0050":0,"taiex":0}')
            I2 = maturity_report.run(pred=pred, picks=picks, out=out, D=D)
            self.assertEqual(I2["new"], [])
            self.assertEqual(json.loads((out / "2026-10-16_h5.json").read_text())["exit"], "x")

    def test_not_matured(self):
        D = _D({"0050": [100] * 8})
        self.assertIsNone(maturity_report.build_one(D, "2026-10-22", "5"))


class Rescore(unittest.TestCase):
    def test_reason_when_not_in_universe(self):
        c = fin_watch.rescore_check({"fin_period": "2026-Q2", "base_ok": False, "exclude_reasons": ["交易日數不足"], "missing_factors": ["股價動能"]},
                                    "2026-Q2", None)
        self.assertTrue(c["fin_in_db"] and c["factors_recomputed"])
        self.assertFalse(c["eligible"])
        self.assertIn("交易日數不足", c["why"])

    def test_scored(self):
        c = fin_watch.rescore_check({"fin_period": "2026-Q2", "base_ok": True, "complete": True,
                                     "ai": {"rec": {"balanced": {"recommendation_eligibility": "normal"}}}}, "2026-Q2", 71.2)
        self.assertTrue(c["eligible"])
        self.assertEqual(c["score"], 71.2)

    def test_missing_row(self):
        self.assertIn("沒有這檔", fin_watch.rescore_check(None, "2026-Q2", None)["why"])


if __name__ == "__main__":
    unittest.main()
