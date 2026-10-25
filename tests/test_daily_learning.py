"""V31 每日自主學習：對帳只做一次、未到期不用、快照不覆寫、有新標籤才重訓、時間不夠安全跳過、同一天重跑不重複。"""
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import daily_learning as dl
from app import ml_research as ml


def fake_D(n_days=40, ids=("0050",) + tuple(f"s{i}" for i in range(40))):
    dates = [f"2026-01-{i + 1:02d}" if i < 31 else f"2026-02-{i - 30:02d}" for i in range(n_days)]
    adj, raw = {}, {}
    rng = np.random.default_rng(2)
    for sid in ids:
        px = 100 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
        df = pd.DataFrame({"open": px, "close": px, "high": px * 1.01, "low": px * 0.99, "volume": 1e6, "value": 1e9}, index=dates)
        adj[sid], raw[sid] = df, df
    return {"dates": dates, "adj": adj, "rawi": raw, "info": {s: {"industry": "電子"} for s in ids}, "idx_s": None}


def sealed(path: Path, sig: str, ids):
    stocks = [{"id": s, "5": {"v21": i, "v21_pct": i / len(ids), "lgbm_pct": 1 - i / len(ids), "ens_pct": 0.5},
               "20": {"v21": i, "v21_pct": i / len(ids), "lgbm_pct": 1 - i / len(ids), "ens_pct": 0.5}} for i, s in enumerate(ids)]
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{sig}.json").write_text(json.dumps({"signal_date": sig, "model_version": "v-test", "generated_at": f"{sig}T15:00:00", "stocks": stocks}), "utf-8")


class Mature(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.D = fake_D()
        self.ids = [f"s{i}" for i in range(40)]

    def test_matures_once_and_never_rewrites(self):
        sealed(self.tmp / "p", self.D["dates"][5], self.ids)
        new = dl.mature(self.D, self.tmp / "p", self.tmp / "m", horizons=("5", "20"))
        self.assertEqual({x["horizon"] for x in new}, {"5", "20"})
        f = self.tmp / "m" / f"{self.D['dates'][5]}_h5.json"
        before = f.read_text("utf-8")
        self.assertEqual(dl.mature(self.D, self.tmp / "p", self.tmp / "m", horizons=("5", "20")), [])   # 重跑不會重複加入
        self.assertEqual(f.read_text("utf-8"), before)

    def test_unmatured_horizon_is_not_used(self):
        sig = self.D["dates"][-10]           # 5 日已到期、20 日還沒
        sealed(self.tmp / "p", sig, self.ids)
        new = dl.mature(self.D, self.tmp / "p", self.tmp / "m", horizons=("5", "20"))
        self.assertEqual([x["horizon"] for x in new], ["5"])
        self.assertFalse((self.tmp / "m" / f"{sig}_h20.json").exists())

    def test_errors_report_counts_top_picks(self):
        for k in (2, 6, 12):
            sealed(self.tmp / "p", self.D["dates"][k], self.ids)
        dl.mature(self.D, self.tmp / "p", self.tmp / "m", horizons=("20",))
        E = dl.errors(dl.load_matured("20", self.tmp / "m"))
        self.assertEqual(E["dates"], 3)
        self.assertEqual(E["models"]["v21"]["dates"], 3)
        self.assertTrue(E["by_industry"]["v21"][0]["n"] == 30 and E["by_industry"]["v21"][0]["enough"])


class Features(unittest.TestCase):
    def test_snapshot_is_not_overwritten_and_drift_flags_missing(self):
        tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(1)
        for i in range(6):
            X = pd.DataFrame({"sid": [f"s{k}" for k in range(300)], "industry": "X", **{f: rng.normal(size=300) for f in ml.FEATS}})
            if i == 5:
                X["roe"] = np.nan          # 今天 ROE 全部缺漏
            self.assertIsNotNone(dl.save_features(X, f"2026-01-0{i + 1}", tmp))
        self.assertIsNone(dl.save_features(X, "2026-01-06", tmp))   # 同一天不覆寫
        d = dl.feature_drift("2026-01-06", tmp)
        self.assertEqual(d["status"], "warn")
        self.assertEqual([r["feature"] for r in d["flagged"]], ["roe"])


class Retrain(unittest.TestCase):
    def test_only_retrain_when_new_labels(self):
        act = {"train_period": {"5": {"through": "2026-01-10"}, "20": {"through": "2026-01-05"}, "60": {"through": None}}}
        self.assertFalse(dl.retrain_decision({"5": "2026-01-10", "20": "2026-01-05", "60": None}, act)[0])   # 同一天第二次排程：不重訓
        need, why = dl.retrain_decision({"5": "2026-01-20", "20": "2026-01-05", "60": None}, act)
        self.assertTrue(need)
        self.assertIn("5 日", why)
        self.assertTrue(dl.retrain_decision({"5": "2026-01-10"}, None)[0])

    def test_refresh_fills_only_matured_labels_without_duplicates(self):
        D = fake_D()
        d = D["dates"]
        panel = pd.DataFrame({"date": [d[5]] * 3 + [d[30]] * 3, "sid": ["s1", "s2", "s3"] * 2, "industry": "X", "v21": 1.0,
                              **{f: 0.0 for f in ml.FEATS},
                              **{f"y{h}": np.nan for h in ml.HORIZONS}, **{f"st{h}": "no_exit" for h in ml.HORIZONS}})
        orig = ml.signal_dates
        ml.signal_dates = lambda dates, start, step=10: [d[5], d[30]]   # 不新增訊號日
        try:
            P, info = dl.refresh_panel(panel, D, {}, log=lambda *a: None)
        finally:
            ml.signal_dates = orig
        self.assertEqual(len(P), 6)
        self.assertEqual(info["new_labels"]["5"], 6)       # d5、d30 的 5 日都已到期
        self.assertEqual(info["new_labels"]["20"], 3)      # 只有 d5 的 20 日到期
        self.assertTrue(P.loc[P["date"] == d[30], "y20"].isna().all())
        self.assertEqual(dl.labeled_through(P)["20"], d[5])

    def test_budget_exhausted_skips_training_safely(self):
        tmp = Path(tempfile.mkdtemp())
        D = fake_D()
        d = D["dates"]
        panel = pd.DataFrame({"date": [d[2]] * 3, "sid": ["s1", "s2", "s3"], "industry": "X", "v21": 1.0, **{f: 0.0 for f in ml.FEATS},
                              **{f"y{h}": 0.01 for h in ml.HORIZONS}, **{f"st{h}": "ok" for h in ml.HORIZONS}})
        panel.to_pickle(tmp / "p.pkl")
        orig = ml.signal_dates
        ml.signal_dates = lambda dates, start, step=10: [d[2]]
        try:
            r = dl.candidate(D, {}, time.time(), 0, log=lambda *a: None, panel_path=tmp / "p.pkl", root=tmp / "models")
        finally:
            ml.signal_dates = orig
        self.assertFalse(r["retrained"])
        self.assertIn("安全跳過", r["reason"])

    def test_no_panel_means_monitoring_only(self):
        r = dl.candidate(fake_D(), {}, time.time(), 999, panel_path=Path(tempfile.mkdtemp()) / "none.pkl")
        self.assertFalse(r["retrained"])


class RunLog(unittest.TestCase):
    def test_failures_are_recorded_and_streak_alerts(self):
        import os
        p = Path(tempfile.mkdtemp()) / "runs.json"
        for i, ok in enumerate((True, False, False)):
            os.environ["GITHUB_RUN_ID"] = f"r{i}"
            dl.record(ok, None if ok else "boom", path=p)
        dl.record(False, "again", path=p)                     # 同一個執行只記一次
        st = dl.run_status(p)
        self.assertEqual(st["runs"], 3)
        self.assertEqual(st["fail_streak"], 2)
        self.assertTrue(st["alert"])
        self.assertIsNotNone(st["last_ok"])
        self.assertIn("boom", st["label"])
        os.environ["GITHUB_RUN_ID"] = "r9"
        dl.record(True, path=p)
        self.assertFalse(dl.run_status(p)["alert"])
        del os.environ["GITHUB_RUN_ID"]

    def test_crash_is_logged_not_reported_as_done(self):
        p = Path(tempfile.mkdtemp()) / "runs.json"
        orig_run, orig_runs = dl.run, dl.RUNS
        dl.run = lambda **k: (_ for _ in ()).throw(RuntimeError("資料庫壞了"))
        dl.RUNS = p
        try:
            self.assertEqual(dl.main([]), 1)
        finally:
            dl.run, dl.RUNS = orig_run, orig_runs
        st = dl.run_status(p)
        self.assertFalse(st["last"]["ok"])
        self.assertIn("資料庫壞了", st["last"]["error"])


if __name__ == "__main__":
    unittest.main()
