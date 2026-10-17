"""V25 持續學習引擎：版本檢查、上線、回滾、新舊模型比賽、錯誤分析。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import learning as L
from app import ml_predict as mp
from app import ml_research as ml
from tests.test_ml_research import WalkForward


def _staging(t: Path, P: pd.DataFrame, name: str) -> Path:
    d = t / name
    mp.export_models(P, use_lgbm=False, out=d)
    return d


def _matured(P, dates):
    """和真實資料一樣：出場日超過資料範圍的標籤是空的。"""
    pos = {d: i for i, d in enumerate(dates)}
    for h in ml.HORIZONS:
        P.loc[P["date"].map(lambda d: pos[d] + 1 + h >= len(dates)), f"y{h}"] = np.nan
    return P


class Registry(unittest.TestCase):
    def setUp(self):
        self.P, self.sigs, self.dates = WalkForward().panel(n_dates=12, n=80, signal=0.05)
        self.P = _matured(self.P, self.dates)
        self.sample = self.P[self.P["date"] == self.sigs[-1]].copy()

    def test_valid_version_goes_live_and_same_hash_is_not_re_registered(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); root = t / "models"
            e = L.register(_staging(t, self.P, "a"), None, self.sample, self.dates, root=root)
            self.assertEqual(e["status"], "active")
            self.assertTrue((root / "ml.json").exists() and (root / "versions" / e["version"] / "ml.json").exists())
            self.assertEqual(json.loads((root / "ml.json").read_text("utf-8"))["version"], e["version"])
            L.register(_staging(t, self.P, "b"), None, self.sample, self.dates, root=root)
            reg = L.load_registry(root / "registry.json")
            self.assertEqual(len(reg["versions"]), 1)
            self.assertEqual(reg["events"][-1]["kind"], "unchanged")

    def test_broken_version_is_rejected_and_old_one_stays_live(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); root = t / "models"
            good = L.register(_staging(t, self.P, "a"), None, self.sample, self.dates, root=root)
            bad = _staging(t, self.P.assign(score_growth=self.P["score_growth"] * -1), "b")
            m = json.loads((bad / "ml.json").read_text("utf-8"))
            m["horizons"]["20"]["ridge"] = [float("nan")] * len(m["horizons"]["20"]["ridge"])
            (bad / "ml.json").write_text(json.dumps(m), "utf-8")
            e = L.register(bad, None, self.sample, self.dates, root=root)
            self.assertEqual(e["status"], "rejected")
            self.assertEqual(L.load_registry(root / "registry.json")["active"], good["version"])
            self.assertEqual(mp.load_models(root)["model_hash"], good["model_hash"])

    def test_unfinished_labels_fail_the_check(self):
        with tempfile.TemporaryDirectory() as t:
            st = _staging(Path(t), self.P, "a")
            cut = self.dates[: self.dates.index(self.sigs[-1]) + 3]  # 資料只到最後訊號日後兩天：20 日標籤不可能到期
            self.assertTrue(any("還沒到期" in p for p in L.validate(st, None, cut)))
            self.assertEqual(L.validate(st, self.sample, self.dates), [])

    def test_rollback_to_previous_version(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); root = t / "models"
            a = L.register(_staging(t, self.P, "a"), None, self.sample, self.dates, root=root)
            P2 = self.P.copy(); P2["y20"] = P2["y20"][::-1].to_numpy()
            b = L.register(_staging(t, P2, "b"), None, self.sample, self.dates, root=root)
            self.assertEqual(b["status"], "active")
            (root / "ml.json").write_text("{", "utf-8")  # 上線版本壞掉
            self.assertEqual(L.rollback("讀不到", root=root), a["version"])
            self.assertEqual(mp.load_models(root)["model_hash"], a["model_hash"])
            reg = L.load_registry(root / "registry.json")
            self.assertEqual({v["version"]: v["status"] for v in reg["versions"]}, {a["version"]: "active", b["version"]: "rolled_back"})


class Gate(unittest.TestCase):
    def daily(self, n, lgbm_ic, lgbm_top, seed=3):
        rng = np.random.default_rng(seed)
        return [{"sig": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", "h": "20", "ver": "x",
                 "ic": {"v21": 0.02 + rng.normal() * 0.03, "lgbm": lgbm_ic + rng.normal() * 0.03},
                 "top": {"v21": 0.005 + rng.normal() * 0.02, "lgbm": lgbm_top + rng.normal() * 0.02}} for i in range(n)]

    def test_accumulating_until_enough_days(self):
        g = L.gate(self.daily(30, 0.2, 0.05), 0.585)
        self.assertEqual(g["status"], "accumulating")

    def test_clearly_better_challenger_passes_but_needs_manual_ok(self):
        g = L.gate(self.daily(80, 0.12, 0.04), 0.585)
        self.assertEqual(g["status"], "passed")
        self.assertIn("人工確認", g["label"])

    def test_same_quality_challenger_is_kept_out(self):
        g = L.gate(self.daily(80, 0.02, 0.005), 0.585)
        self.assertEqual(g["status"], "kept")


class Errors(unittest.TestCase):
    def test_groups_cover_regime_buckets_and_flag_evidence(self):
        P, sigs, dates = WalkForward().panel(n_dates=14, n=80, signal=0.05)
        wf = ml.walk_forward(P, 20, sigs, dates, use_lgbm=False)
        pr = wf["preds"].rename(columns={"ridge": "lgbm"})
        idx = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
        E = L.error_analysis(pr, P, 20, idx)
        dims = {g["dim"] for g in E["groups"]}
        self.assertTrue({"市場環境", "波動", "近 3 月漲幅", "估值（盈餘殖利率）", "規模"} <= dims)
        self.assertTrue(all(g["evidence"] in ("IC 估計可靠（|t|≥2）", "樣本不足，僅供觀察") for g in E["groups"]))


class Lessons(unittest.TestCase):
    def test_importance_change_is_reported_without_causal_claims(self):
        a = {f: 1.0 for f in ml.FEATS}; a["ret_3m"] = 30.0
        b = dict(a); b["ret_3m"] = 0.5; b["vol_1y"] = 40.0
        reg = {"versions": [{"status": "previous", "importance": {"20": a}}, {"status": "active", "importance": {"20": b}}]}
        txt = "".join(L.lessons(reg))
        self.assertIn("一年波動", txt)
        self.assertIn("近 3 月報酬", txt)
        self.assertIn("不代表因果", txt)


if __name__ == "__main__":
    unittest.main()
