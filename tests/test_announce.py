"""2026-11-20 公告後股價反應研究與影子紀錄：訊號日判斷、驚喜分數只用過去資料、成交與成本、封存不覆寫。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import announce_shadow as sh
from app import announce_study as A


def _D(n=30, sids=("1101", "1102")):
    dates = [str(d.date()) for d in pd.bdate_range("2026-11-02", periods=n)]
    O = pd.DataFrame({s: np.linspace(100, 110, n) for s in sids}, index=dates)
    b = pd.Series(np.linspace(100, 101, n), index=dates)
    return {"O": O, "C": O.copy(), "dates": dates, "bo": b, "bc": b}


class Study(unittest.TestCase):
    def test_surge_uses_only_past_months(self):
        ym = pd.period_range("2024-01", "2025-12", freq="M")
        rev = pd.DataFrame({"stock_id": "1101", "ym": [str(p) for p in ym], "revenue": [100 + (i * 7) % 5 for i in range(len(ym))]})
        rev.loc[rev["ym"] == "2025-12", "revenue"] = 300          # 最後一個月暴增
        S = A.surge_table(rev)
        self.assertGreater(S.loc[pd.Period("2025-12", "M"), "1101"], 3)
        self.assertTrue(np.isnan(S.loc[pd.Period("2024-12", "M"), "1101"]))   # 歷史不到 9 個月不算

    def test_hold_cost_and_unfillable(self):
        D = _D()
        D["O"].iloc[1, 1] = np.nan                                # 1102 隔天沒成交：買不到
        r = A.hold(D, ["1101", "1102"], 1, 21)
        self.assertEqual(r["n"], 1)
        gross = D["O"]["1101"].iloc[21] / D["O"]["1101"].iloc[1] - 1
        self.assertAlmostEqual(r["net"], gross - A.COST)
        self.assertAlmostEqual(r["excess"], r["net"] - (D["bo"].iloc[21] / D["bo"].iloc[1] - 1))

    def test_judge_gate(self):
        ev = [{"signal": f"{2019 + k // 12}-{k % 12 + 1:02d}-11", "R2": {"excess": 0.03 + (0.01 if k % 2 else -0.01), "net": 0.04, "bench": 0.01, "big_loss": 0}}
              for k in range(72)]
        ev += [{"signal": "2025-03-11", "R2": {"excess": 0.01, "net": 0.02, "bench": 0.01, "big_loss": 0}}]
        v = A.judge(ev, "R2")
        self.assertTrue(v["passed"])
        ev[-1]["R2"]["excess"] = -0.01                           # 保留期輸 0050 就否決
        self.assertFalse(A.judge(ev, "R2")["passed"])


class Shadow(unittest.TestCase):
    def test_due_today(self):
        self.assertEqual(sh.due_today(["2026-12-10", "2026-12-11"]), {"R2": "2026-11"})
        self.assertEqual(sh.due_today(["2026-12-11", "2026-12-14"]), {})
        self.assertEqual(sh.due_today(["2027-01-08", "2027-01-11"]), {"R2": "2026-12"})
        self.assertEqual(sh.due_today(["2026-11-13", "2026-11-16"]), {"Q1": "2026-Q3"})

    def test_seal_never_overwrites(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            D = _D()
            f = out / f"{D['dates'][-1]}.json"
            f.write_text(json.dumps({"signal_date": D["dates"][-1], "rules": {"R2": {"picks": ["1101"]}}}))
            self.assertEqual(sh.seal(D, {"R2": "2026-11"}, out), [])
            self.assertEqual(json.loads(f.read_text())["rules"]["R2"]["picks"], ["1101"])

    def test_score_matures_q1_after_20_days(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            D = _D(n=30)
            (out / f"{D['dates'][2]}.json").write_text(json.dumps({"signal_date": D["dates"][2], "rules": {"Q1": {"picks": ["1101"]}}}))
            (out / f"{D['dates'][20]}.json").write_text(json.dumps({"signal_date": D["dates"][20], "rules": {"Q1": {"picks": ["1101"]}}}))
            r = sh.score(D, out)
            self.assertEqual(r["summary"]["Q1"]["sealed"], 2)
            self.assertEqual(r["summary"]["Q1"]["matured"], 1)
            self.assertFalse(r["summary"]["Q1"]["enough"])


if __name__ == "__main__":
    unittest.main()
