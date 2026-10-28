"""V21 檢討：持有期（交易日）定義、績效口徑一致、加權指數同期比較、事前登記研究的挑股規則。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import pit_backtest as pb
from app import selection_study as ss

# 2026/10 真實交易日：10/23 颱風停市、10/26 光復節補假，兩天都沒有行情
OCT = ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22",
       "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"]


class Window(unittest.TestCase):
    def test_holiday_and_typhoon(self):
        # 10/16 訊號 → 10/19 開盤進場 → 5 個交易日（10/19、20、21、22、27）後 10/28 開盤出場
        self.assertEqual(pb.window(OCT, "2026-10-16", 5), ("2026-10-19", "2026-10-28"))
        self.assertEqual(pb.window(OCT, "2026-10-22", 1), ("2026-10-27", "2026-10-28"))

    def test_not_matured_and_unknown(self):
        self.assertIsNone(pb.window(OCT, "2026-10-28", 5))
        self.assertIsNone(pb.window(OCT, "2026-10-23", 1))   # 停市日不是訊號日

    def test_weekend_and_year_end(self):
        d = ["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06"]
        self.assertEqual(pb.window(d, "2025-12-31", 2), ("2026-01-02", "2026-01-06"))
        self.assertEqual(pb.window(d, "2025-12-30", 1), ("2025-12-31", "2026-01-02"))

    def test_selection_periods_do_not_overlap(self):
        d = [f"d{i:03d}" for i in range(100)]
        ps = ss.signal_dates(d, start="d010", step=20)
        self.assertTrue(all(a[2] <= b[1] for a, b in zip(ps, ps[1:])))
        self.assertTrue(all(d.index(p[2]) - d.index(p[1]) == 20 for p in ps))


def _D():
    idx = pd.Index(OCT)
    px = pd.DataFrame({"stock_id": "0050", "open": np.linspace(100, 110, len(OCT)), "close": np.linspace(101, 111, len(OCT)),
                       "high": 0, "low": 0, "volume": 1, "value": 1}, index=idx)
    return {"dates": OCT, "adj": {"0050": px}, "idx_s": pd.Series(np.linspace(1000, 1100, len(OCT)), index=idx)}


class TaiexWindow(unittest.TestCase):
    def test_same_close_window(self):
        from app import maturity_report as mr
        t = mr._taiex(_D(), "2026-10-16", "5")
        self.assertEqual(t["window"], ["2026-10-16", "2026-10-27"])
        c = pb.close_return(_D(), "0050", "2026-10-16", "2026-10-27")
        self.assertAlmostEqual(t["bench_0050_close"], round(c * 100, 2))

    def test_correction_is_append_only(self):
        from app import maturity_report as mr
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            orig = {"signal_date": "2026-10-16", "horizon": 5, "taiex": -2.15, "bench_0050": 0.84}
            (out / "2026-10-16_h5.json").write_text(json.dumps(orig), "utf-8")
            a = mr.corrections(_D(), out)
            b = mr.corrections(_D(), out)
            self.assertEqual(len(a), 1)
            self.assertEqual(len(b), 1)   # 不重複附加
            self.assertEqual(json.loads((out / "2026-10-16_h5.json").read_text("utf-8")), orig)   # 原檔不動
            self.assertEqual(a[0]["original"]["taiex"], -2.15)


class Picks(unittest.TestCase):
    def setUp(self):
        n = 40
        self.df = pd.DataFrame({"rk_aggressive": range(1, n + 1), "sc_aggressive": np.linspace(90, 50, n),
                                "industry": ["A"] * 5 + ["B"] * (n - 5), "market_cap": np.arange(n)[::-1] * 1e9,
                                "dist_ma60": [1] * n, "ma60_slope_20": [1] * n, "growth": [70] * n,
                                "ret": [0.01] * n}, index=[f"s{i}" for i in range(n)])

    def test_buffer_keeps_holdings(self):
        prev = ["s25", "s0"]
        pk = ss.pick(self.df, "aggressive", "BUF", prev)
        self.assertEqual(pk[:2], ["s25", "s0"])
        self.assertEqual(len(pk), 10)
        self.assertNotIn("s35", ss.pick(self.df, "aggressive", "BUF", ["s35"]))   # 掉出前 30 名就換掉

    def test_industry_cap(self):
        pk = ss.pick(self.df, "aggressive", "IND2")
        self.assertEqual(sum(s in pk for s in ["s0", "s1", "s2", "s3", "s4"]), 2)

    def test_cost_only_on_new(self):
        P = [{"df": self.df}, {"df": self.df}]
        r = ss._net(P, [["s0", "s1"], ["s0", "s2"]])
        self.assertAlmostEqual(r[0], 0.01 - pb.COST)
        self.assertAlmostEqual(r[1], 0.01 - pb.COST / 2)


if __name__ == "__main__":
    unittest.main()
