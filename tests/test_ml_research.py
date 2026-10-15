"""V22：硬性條件缺值不能當成通過；預測模型研究的時間切分不能偷看未來。"""
import unittest

import numpy as np
import pandas as pd

from app import ai_model, ml_research as ml


class HardConditions(unittest.TestCase):
    def row(self, **kw):
        r = {"base_ok": True, "eps_ttm": 2, "ocf_ttm": 1, "debt_ratio": 40, "vol_1y_pct": 50, "avg_value_20": 1e8, "complete": True,
             "rev_yoy": 5, "close": 10, "ma60": 9, "is_financial": False}
        r.update(kw)
        return r

    def test_missing_value_is_not_pass(self):
        for mode in ai_model.MODES:
            self.assertTrue(ai_model.eligible(self.row(), mode), mode)
        self.assertFalse(ai_model.eligible(self.row(ocf_ttm=None), "stable"))      # 缺現金流
        self.assertFalse(ai_model.eligible(self.row(avg_value_20=None), "stable"))  # 缺成交金額
        self.assertFalse(ai_model.eligible(self.row(eps_ttm=None), "balanced"))
        self.assertFalse(ai_model.eligible(self.row(ma60=None), "aggressive"))

    def test_not_applicable_only_for_financial_debt_ratio(self):
        # 金融業負債比本來就高，這一項「不適用」；其他條件照常要求
        self.assertTrue(ai_model.eligible(self.row(is_financial=True, debt_ratio=90), "stable"))
        self.assertFalse(ai_model.eligible(self.row(is_financial=False, debt_ratio=90), "stable"))
        self.assertFalse(ai_model.eligible(self.row(is_financial=True, ocf_ttm=None), "stable"))
        na = [c for m in ai_model.MODES.values() for c in m["hard"] if c.get("skip_financial")]
        self.assertEqual([c["m"] for c in na], ["debt_ratio"])


class WalkForward(unittest.TestCase):
    def panel(self, n_dates=14, n=60, signal=0.0, seed=1):
        rng = np.random.default_rng(seed)
        dates = [f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_dates * ml.STEP + 30)]
        sigs = ml.signal_dates(dates, dates[0])[:n_dates]
        rows = []
        for s in sigs:
            x = rng.normal(size=n)
            for i in range(n):
                r = {"date": s, "sid": f"{1000 + i}", "industry": "x", "v21": float(rng.normal())}
                for f in ml.FEATS:
                    r[f] = float(rng.normal())
                r["score_growth"] = float(x[i])
                for h in ml.HORIZONS:
                    r[f"y{h}"] = signal * x[i] + float(rng.normal()) * 0.05
                    r[f"st{h}"] = "ok"
                rows.append(r)
        return pd.DataFrame(rows), sigs, dates

    def test_purge_never_trains_on_unfinished_labels(self):
        P, sigs, dates = self.panel()
        wf = ml.walk_forward(P, 20, sigs, dates, use_lgbm=False)
        pos = {d: i for i, d in enumerate(dates)}
        for t in wf["tests"]:
            trained = [s for s in sigs if pos[s] + 1 + 20 < pos[t]]
            self.assertGreaterEqual(len(trained), ml.MIN_TRAIN)
            self.assertTrue(all(s < t for s in trained))
        self.assertNotIn(sigs[ml.MIN_TRAIN], wf["tests"])  # 20 日標籤要兩個訊號日後才確定，第 9 個訊號日還不能測

    def test_learns_real_signal_and_not_noise(self):
        P, sigs, dates = self.panel(signal=0.05)
        wf = ml.walk_forward(P, 5, sigs, dates, use_lgbm=False)
        ev = ml.evaluate(wf["preds"], 5, ["v21", "ridge", "logit"])
        self.assertGreater(ev["ridge"]["ic"], 0.3)
        self.assertLess(abs(ev["v21"]["ic"]), 0.15)
        P0, *_ = self.panel(signal=0.0, seed=3)
        ev0 = ml.evaluate(ml.walk_forward(P0, 5, sigs, dates, use_lgbm=False)["preds"], 5, ["ridge"])
        self.assertLess(abs(ev0["ridge"]["ic"]), 0.1)

    def test_sufficiency(self):
        s = ml.sufficiency({"ic_sd": 0.12}, 30, 20)   # 300 個交易日 = 15 個獨立 20 日期間
        self.assertEqual(s["independent_periods"], 15.0)
        self.assertAlmostEqual(s["min_detectable_ic"], round(2.8 * 0.12 / 15 ** 0.5, 4))
        self.assertFalse(s["enough"])
        self.assertGreater(s["years_needed_for_0.03"], 5)


if __name__ == "__main__":
    unittest.main()
