"""V30 LightGBM 可信度檢查：去掉最賺持股、最好一期、市場環境不用未來資料、分歧與成本計算、退步紀錄格式。"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import model_race as mr
from app import robustness as rb


def fake_preds(n_dates=12, n=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dates):
        d = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        for k in range(n):
            y = rng.normal(0, 0.05)
            rows.append({"date": d, "sid": f"s{k}", "y20": y, "v21": rng.normal(), "lgbm": y + rng.normal(0, 0.05), "market_cap": float(k)})
    return pd.DataFrame(rows)


class Trim(unittest.TestCase):
    def test_removing_one_huge_winner_shows_dependence(self):
        h = pd.DataFrame({"date": ["d1"] * 10 + ["d2"] * 10, "sid": [f"s{i}" for i in range(20)],
                          "y": [0.0] * 19 + [2.0], "cost": [0.0] * 20})
        r = rb.trim_study(h)
        self.assertAlmostEqual(r["all"], 10.0)          # (0 + 20%) / 2
        self.assertAlmostEqual(r["no_top1"], 0.0)       # 去掉那一檔就沒了
        self.assertAlmostEqual(r["no_best_period"], 0.0)
        self.assertEqual(r["top_contrib"][0]["sid"], "s19")

    def test_holdings_match_race_portfolio(self):
        P = fake_preds()
        use = sorted(P["date"].unique())
        h = rb.holdings(P, "lgbm", use)
        p = mr.portfolio(P, "lgbm", use, {})
        self.assertAlmostEqual(rb.trim_study(h)["all"], p["net_excess"], places=2)
        self.assertAlmostEqual(rb.cost_study(h)["x1"], p["net_excess"], places=2)
        self.assertLess(rb.cost_study(h)["x3"], rb.cost_study(h)["x1"])


class Regime(unittest.TestCase):
    def test_vol_threshold_uses_only_past(self):
        idx = pd.Series(np.linspace(100, 200, 300), index=[f"d{i:03d}" for i in range(300)])
        a = rb.regime4(idx)
        idx2 = idx.copy()
        idx2.iloc[250:] = idx2.iloc[250:] * np.random.default_rng(3).uniform(0.5, 1.5, 50)   # 改掉未來
        b = rb.regime4(idx2)
        self.assertEqual({k: v for k, v in a.items() if k < "d250"}, {k: v for k, v in b.items() if k < "d250"})
        self.assertTrue(all(v.startswith("up") for v in a.values()))


class Study(unittest.TestCase):
    def test_run_produces_all_sections(self):
        P = fake_preds()
        tests = sorted(P["date"].unique())
        r = rb.run(P, tests, tests[8], None, [(tests[0], {f: 1.0 for f in rb.GROUPS["price"][1]})], mr.portfolio, mr.ic_stats)
        for k in ("trim", "trim_holdout", "cost", "size", "disagreement", "importance", "findings"):
            self.assertIn(k, r)
        self.assertEqual(set(r["size"]), set(rb.SIZE))
        self.assertEqual(r["importance"]["2026"]["price"], 100.0)
        self.assertIsNotNone(r["disagreement"]["agree"]["n"])

    def test_groups_cover_every_feature_once(self):
        from app import ml_research as ml
        feats = [f for _, (_, fs) in rb.GROUPS.items() for f in fs]
        self.assertEqual(sorted(feats), sorted(ml.FEATS))


class ExperimentLog(unittest.TestCase):
    def test_every_item_has_required_fields(self):
        E = json.loads((Path(__file__).resolve().parent.parent / "history" / "research" / "experiments.json").read_text("utf-8"))
        need = {"date", "version", "change", "hypothesis", "improved", "worse", "result", "cause", "decision", "evidence"}
        for it in E["items"]:
            self.assertTrue(need <= set(it), it.get("change"))
            self.assertIn(it["decision"], E["decisions"])


class WalkAblation(unittest.TestCase):
    def test_walk_trains_ablated_models_and_collects_importance(self):
        try:
            import lightgbm  # noqa: F401
        except Exception:
            self.skipTest("沒有安裝 LightGBM")
        from app import ml_research as ml
        rng = np.random.default_rng(5)
        dates = [f"d{i:03d}" for i in range(200)]
        sigs = dates[:160:10]
        rows = []
        for d in sigs:
            for k in range(150):
                x = {"date": d, "sid": f"s{k}", "industry": "X", "v21": rng.normal(), "y20": rng.normal(0, 0.05)}
                x.update({f: rng.normal() for f in ml.FEATS})
                rows.append(x)
        imp = []
        preds, tests = mr.walk(pd.DataFrame(rows), sigs, dates, None, True, ablate={g: f for g, (_, f) in rb.GROUPS.items()}, imp=imp)
        self.assertTrue(tests)
        for g in rb.GROUPS:
            self.assertIn(f"abl_{g}", preds)
        self.assertEqual(len(imp), len(tests))
        r = rb.run(preds, tests, tests[len(tests) // 2], None, imp, mr.portfolio, mr.ic_stats)
        self.assertEqual(set(r["ablation"]) - {"lgbm"}, set(rb.GROUPS))


if __name__ == "__main__":
    unittest.main()
