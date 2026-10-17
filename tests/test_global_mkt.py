import unittest
from datetime import date

from app import global_mkt

SOX = [{"date": d, "Close": c} for d, c in [("2026-10-05", 13172.74), ("2026-10-06", 13217.82), ("2026-10-07", 13066.15),
                                             ("2026-10-08", 12623.71), ("2026-10-09", 12572.42), ("2026-10-12", 12236.79)]]


class GlobalMarkets(unittest.TestCase):
    def test_summary_matches_finmind_rows(self):
        s = global_mkt.summarize("^SOX", "費城半導體指數", SOX)
        self.assertEqual(s["date"], "2026-10-12")
        self.assertEqual(s["chg_pct"], -2.67)   # 12236.79 / 12572.42 - 1
        self.assertEqual(s["streak"], -4)       # 10/07、10/08、10/09、10/12 連跌
        self.assertIsNotNone(s["ret_5d"])

    def test_failures_become_warnings(self):
        def get(ds, sid, start):
            if sid == "^SOX":
                return SOX
            if ds == "TaiwanExchangeRate":
                return [{"date": "2026-10-12", "spot_buy": 31.9, "spot_sell": 32.0}, {"date": "2026-10-13", "spot_buy": 32.1, "spot_sell": 32.2}]
            raise RuntimeError("boom")
        r = global_mkt.build(get, date(2026, 10, 13))
        self.assertEqual([m["symbol"] for m in r["markets"]], ["^SOX"])
        self.assertEqual(len(r["warnings"]), len(global_mkt.SYMBOLS) - 1 + 1)  # 另外一則是原油
        self.assertIsNone(r["oil"])
        self.assertEqual(r["fx_usd"]["chg"], 0.2)


if __name__ == "__main__":
    unittest.main()
