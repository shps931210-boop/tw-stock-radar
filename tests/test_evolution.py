"""V14／V15：目標記憶、實驗記憶不重複、研究代理把關、反省不假裝學習、研究任務只增不刪、五道晉升關卡、
每日流程三種狀態、三本帳的真實前瞻指標、投資組合研究的選股規則。"""
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from app import daily_learning as dl
from app import evolution as ev
from app import ledgers
from app import model_race as mr
from app import portfolio_study as ps

ROOT = Path(__file__).resolve().parent.parent


class Mission(unittest.TestCase):
    def test_goals_and_whitelist(self):
        m = ev.load_mission()
        ids = [g["id"] for g in m["goals"]]
        self.assertEqual(ids[:7], ["G1", "G2", "G3", "G4", "G5", "G6", "G7"])
        self.assertTrue(all(g["priority"] in ("P0", "P1", "P2") for g in m["goals"]))
        self.assertNotIn("deploy", m["research_whitelist"])
        self.assertTrue(m["promotion"]["human_approval"])
        self.assertFalse(m["promotion"]["auto_promote"])

    def test_change_without_version_is_flagged(self):
        tmp = Path(tempfile.mkdtemp())
        f, log = tmp / "m.yaml", tmp / "log.json"
        f.write_text("version: 1\ngoals: []\n", "utf-8")
        self.assertEqual(ev.track_mission(ev.load_mission(f), log)["status"], "first")
        self.assertEqual(ev.track_mission(ev.load_mission(f), log)["status"], "unchanged")
        f.write_text("version: 1\ngoals: [{id: G1}]\n", "utf-8")
        self.assertTrue(ev.track_mission(ev.load_mission(f), log)["alert"])
        f.write_text("version: 2\ngoals: [{id: G1}]\n", "utf-8")
        self.assertEqual(ev.track_mission(ev.load_mission(f), log)["status"], "changed")
        self.assertEqual(len(json.loads(log.read_text("utf-8"))["entries"]), 3)

    def test_code_never_writes_mission(self):
        for p in (ROOT / "app").glob("*.py"):
            src = p.read_text("utf-8")
            self.assertNotIn("_write(MISSION", src, p.name)
            self.assertNotIn("MISSION.write", src, p.name)


def exp(spec="w_25", data_end="2026-09-11", passed=False, periods=82, run_at="2026-10-25T10:00:00"):
    return {"run_at": run_at, "family": "lgbm_study", "spec": spec, "data_end": data_end, "periods": periods, "passed": passed, "goal": "G1"}


class ExperimentMemory(unittest.TestCase):
    def setUp(self):
        self.p = Path(tempfile.mkdtemp()) / "experiments.json"
        self.p.write_text(json.dumps({"items": [{"date": "2026-10-22", "change": "人工紀錄"}]}), "utf-8")

    def test_same_experiment_registered_once_and_failures_kept(self):
        a, new1 = ev.register(exp(), self.p)
        b, new2 = ev.register(exp(), self.p)
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(a["id"], b["id"])
        c, _ = ev.register(exp(data_end="2026-10-09", periods=83), self.p)
        self.assertEqual(c["related"], [a["id"]])
        self.assertEqual(c["repeat_of_failed"], 1)
        E = json.loads(self.p.read_text("utf-8"))
        self.assertEqual(len(E["items"]), 1)   # 人工整理的決定不被動到
        self.assertEqual(len(E["auto"]), 2)    # 失敗的實驗也留著

    def test_agent_skips_repeated_failures_without_new_data(self):
        ev.register(exp(data_end="2026-09-11", periods=82), self.p)
        ev.register(exp(data_end="2026-09-25", periods=83), self.p)
        self.assertFalse(ev.allowed("lgbm_study", periods_now=84, path=self.p)["run"])
        self.assertTrue(ev.allowed("lgbm_study", periods_now=90, path=self.p)["run"])
        self.assertTrue(ev.allowed("portfolio_study", periods_now=84, path=self.p)["run"])

    def test_ingest_studies_once(self):
        bt = Path(tempfile.mkdtemp())
        (bt / "lgbm_study.json").write_text(json.dumps({"study_version": "x", "generated_at": "2026-10-25T10:00:00", "last_test": "2026-09-11",
            "independent_periods": 82, "t_need": 2.69, "results": {"base": {"portfolio": {"net_excess": -0.6}},
            "w_25": {"name": "最近 25", "portfolio": {"net_excess": -0.2}, "ic": {"ic": 0.08}, "vs_base": {"diff": 0.4, "t": 0.8},
                     "verdict": {"passed": False, "label": "方向較好但證據不足"}}}}), "utf-8")
        self.assertEqual(len(ev.ingest_studies(bt, self.p)), 1)
        self.assertEqual(len(ev.ingest_studies(bt, self.p)), 0)


def report(matured=0, labels=0, gate=None):
    return {"date": "2026-10-27", "matured_today": [{"signal_date": "2026-09-26", "horizon": "20", "stocks": matured}] if matured else [],
            "matured_stocks_today": matured, "candidate": {"retrained": False, "reason": "沒有新標籤", "new_labels": {"20": labels} if labels else {}},
            "errors": {"dates": 0}, "gate": gate or {"status": "accumulating", "label": "累積中", "days": 0, "need": 60, "criteria": []},
            "feature_effect": {"hypotheses": ["rev_yoy 最近的實盤效果比研究時弱"]}, "data": {"ok": True}, "sealed": {"today": True}}


class Reflection(unittest.TestCase):
    def test_no_new_labels_means_no_supervised_learning(self):
        r = ev.reflect(report())
        self.assertFalse(r["supervised_learning"])
        self.assertIn("沒有新的監督式學習", r["supervised_text"])
        self.assertFalse(any(x["done"] for x in r["q5_outcome"] if x["kind"] in ("新增資料", "訓練候選模型", "驗證候選模型改善")))
        self.assertEqual(r["q4_hypotheses"]["status"], "待驗證假設")
        self.assertEqual(r["q3_degrading"]["status"], "accumulating")

    def test_new_labels_count_as_learning_data(self):
        r = ev.reflect(report(matured=300))
        self.assertTrue(r["supervised_learning"])

    def test_degrading_needs_enough_samples(self):
        rows = [{"date": f"d{i}", "matured_dates": 30, "models": {"v21": {"ic": 0.05 - i * 0.001, "top_net": 1.0 - i * 0.05}}} for i in range(25)]
        t = ev._trend(rows)
        self.assertEqual(t["status"], "warn")
        self.assertEqual(ev._trend(rows[:1])["status"], "accumulating")


class Tasks(unittest.TestCase):
    def test_tasks_grow_only_and_rerun_is_idempotent(self):
        tmp = Path(tempfile.mkdtemp())
        q = {"items": [{"kind": "industry", "key": "lgbm:航運", "text": "航運拖累", "status": "待驗證", "times": 3, "first_seen": "2026-10-27"},
                       {"kind": "data", "key": "rev_yoy", "text": "rev_yoy 分布改變", "status": "待驗證", "times": 1, "first_seen": "2026-10-27"}]}
        T1 = ev.build_tasks(report(), q, tmp / "t.json", tmp / "e.json", tmp)
        T2 = ev.build_tasks(report(), q, tmp / "t.json", tmp / "e.json", tmp)
        self.assertEqual(len(T1["tasks"]), 3)   # 兩項錯誤記憶＋一項特徵失效研究模板
        self.assertEqual([t["id"] for t in T2["tasks"]], [t["id"] for t in T1["tasks"]])
        t = {x["key"]: x for x in T2["tasks"]}
        self.assertTrue(t["q:industry:lgbm:航運"]["allowed"])
        self.assertTrue(t["q:data:rev_yoy"]["needs_human"])    # 資料問題不在白名單：要人工
        for k in ("question", "goal", "hypothesis", "metric", "data_needed", "design", "oos", "budget", "stop_rule", "priority"):
            self.assertIn(k, t["q:industry:lgbm:航運"])
        T3 = ev.build_tasks(report(), {"items": []}, tmp / "t.json", tmp / "e.json", tmp)
        self.assertEqual(len(T3["tasks"]), 3)   # 不會刪除


def gate(ok_ic=True, ok_top=True, ok_beat=True, ok_mdd=True, days=60):
    return {"status": "passed" if all((ok_ic, ok_top, ok_beat, ok_mdd)) and days >= 60 else "kept", "label": "x", "days": days, "need": 60,
            "criteria": [{"name": "Rank IC 勝過 V21（Newey-West t）", "ok": ok_ic, "value": ""}, {"name": "前 20 名扣成本超額報酬", "ok": ok_top, "value": ""},
                         {"name": "跑贏 0050 的天數比例（扣成本）", "ok": ok_beat, "value": ""}, {"name": "最大回撤不比 V21 差太多", "ok": ok_mdd, "value": ""}]}


class Governance(unittest.TestCase):
    def setUp(self):
        self.bt = Path(tempfile.mkdtemp())
        (self.bt / "audit.json").write_text(json.dumps({"passed": 6, "total": 6}), "utf-8")

    def test_all_gates_only_recommend(self):
        g = ev.governance({"gate": gate()}, self.bt)
        self.assertTrue(g["recommend"])
        self.assertTrue(g["human_approval"])
        self.assertIn("等網站主人核准", g["label"])

    def test_regression_or_thin_evidence_is_rejected(self):
        self.assertFalse(ev.governance({"gate": gate(ok_ic=False)}, self.bt)["recommend"])        # 候選模型退步
        self.assertFalse(ev.governance({"gate": gate(days=20)}, self.bt)["recommend"])            # 改善但前瞻樣本不足
        self.assertFalse(ev.governance({"gate": gate(ok_mdd=False)}, self.bt)["recommend"])       # 風險變大
        (self.bt / "audit.json").write_text(json.dumps({"passed": 5, "total": 6}), "utf-8")
        self.assertFalse(ev.governance({"gate": gate()}, self.bt)["recommend"])                  # 資料稽核沒過


class Pipeline(unittest.TestCase):
    def test_three_states_with_reasons(self):
        rep = {**report(), "data": {"ok": False, "why": "上櫃資料只有 40%"}, "sealed": {"today": False}}
        P = dl.pipeline(rep, {"錯誤分析": "ValueError: x"}, {"steps": [{"step": "更新研究任務", "status": "success"}]}, True)
        st = {x["step"]: x for x in P}
        self.assertEqual(st["資料品質檢查"]["status"], "failed")
        self.assertIn("40%", st["資料品質檢查"]["why"])
        self.assertEqual(st["預測封存"]["status"], "skipped")
        self.assertEqual(st["錯誤分析"]["status"], "failed")
        self.assertEqual(st["研究任務更新"]["status"], "success")
        self.assertEqual(st["晉升資格評估"]["status"], "skipped")
        self.assertTrue(all(x["status"] in ("success", "failed", "skipped") for x in P))

    def test_evolve_survives_step_failure(self):
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        with mock.patch.object(ev, "EXPERIMENTS", tmp / "e.json"), mock.patch.object(ev, "MISSION_LOG", tmp / "m.json"), \
                mock.patch.object(ev, "TASKS", tmp / "t.json"), mock.patch.object(ev, "BT", tmp):
            out = ev.evolve({"date": "2026-10-27"}, None, None)   # 報告欄位不齊：失敗的步驟記下來，其他照跑
        self.assertTrue(any(s["status"] == "failed" for s in out["steps"]) or out["reflection"] is not None)
        self.assertIn("steps", out)


class LiveLedger(unittest.TestCase):
    def test_live_metrics_from_matured(self):
        rng = np.random.default_rng(1)
        L = []
        for k in range(25):
            ids = [f"s{i}" for i in range(50)]
            y = rng.normal(0, 0.05, 50)
            L.append({"signal_date": f"2026-0{1 + k // 10}-{k % 10 + 10}", "ids": ids, "industry": ["電子"] * 50, "excess": list(y),
                      "pct": {"v21": list(np.argsort(np.argsort(y)) / 50), "lgbm": list(rng.random(50)), "ens": list(rng.random(50))}})
        m = ledgers.live_metrics(L)
        self.assertEqual(m["matured_dates"], 25)
        self.assertGreater(m["models"]["v21"]["ic"], 0.9)        # 完美排序
        self.assertGreater(m["models"]["v21"]["top_net"], 0)
        self.assertEqual(m["models"]["v21"]["hit"], 100.0)
        self.assertEqual(ledgers.live_metrics([])["matured_dates"], 0)


def fake_preds(n_dates=40, n=60, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for i in range(n):
            rows.append({"date": f"d{d:03d}", "sid": f"s{i}", "industry": f"ind{i % 6}", "base": rng.random(), "v21": rng.random(), "w_25": rng.random(),
                         "y20": rng.normal(0, 0.05), "y60": rng.normal(0, 0.08), "vol_1y": rng.uniform(10, 60), "avg_value_20": np.log(rng.uniform(1e6, 1e9))})
    return pd.DataFrame(rows)


class PortfolioRules(unittest.TestCase):
    def setUp(self):
        self.P = fake_preds()
        self.dates = sorted(self.P["date"].unique())
        self.B = {d: (0.01, 0.01) for d in self.dates}

    def test_top10_matches_model_race(self):
        a = ps.book(self.P, "top10", self.dates, self.B)
        b = mr.portfolio(self.P, "base", self.dates, self.B)
        self.assertAlmostEqual(a["net_excess"], b["net_excess"], places=2)

    def test_rules(self):
        g = self.P[self.P["date"] == "d000"]
        w = ps.select(g, "ind_cap", None)
        self.assertLessEqual(g.set_index("sid").loc[w.index, "industry"].value_counts().max(), 2)
        self.assertEqual(len(ps.select(g, "top5", None)), 5)
        el = ps.select(g, "eligible", None)
        self.assertTrue((g.set_index("sid").loc[el.index, "avg_value_20"] >= np.log(ps.MIN_VALUE)).all())
        for v in ("uncert", "liq_w", "inv_vol"):
            self.assertAlmostEqual(float(ps.select(g, v, None).sum()), 1.0)
        prev = ps.select(g, "top10", None)
        g2 = g.assign(base=g["base"].sample(frac=1, random_state=1).to_numpy())
        top30 = set(g2.nlargest(30, "base")["sid"])
        kept = ps.select(g2, "buffer", prev)
        self.assertTrue(set(prev.index) & top30 <= set(kept.index))

    def test_walk_select_uses_only_past(self):
        R = {"top10": {"dates": self.dates, "series": [0.0] * 40}, "a": {"dates": self.dates, "series": [1.0] * 20 + [-1.0] * 20},
             "b": {"dates": self.dates, "series": [-1.0] * 20 + [5.0] * 20}}
        s = ps.walk_select(R, ["top10", "a", "b"])
        self.assertEqual(s["periods"], 40 - ps.MIN_PAST - 1)
        # 第 21 期時只看得到之前的資料：「a」領先，所以會先選到 a，吃到 a 後半段的 -1
        self.assertIn("a", s["picked"])


if __name__ == "__main__":
    unittest.main()


class V16(unittest.TestCase):
    def test_templates_create_full_tasks(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "portfolio_study.json").write_text(json.dumps({"regimes": {"top10": {"up_calm": {"name": "多頭・平穩", "periods": 24, "net_excess": -1.87, "t": -2.32}}},
            "loss_attribution": {"features": [{"feature": "vol_1y", "name": "一年波動", "loss_pct": 51.2, "other_pct": 38.3, "t": 3.58}]}}), "utf-8")
        T = ev.build_tasks(report(), None, tmp / "t.json", tmp / "e.json", tmp)
        by = {t.get("template"): t for t in T["tasks"]}
        for k in ("regime", "big_loss", "feature_failure"):
            self.assertIn(k, by)
            for f in ("hypothesis", "data_needed", "design", "metric", "oos", "budget", "stop_rule"):
                self.assertTrue(by[k][f], (k, f))
            self.assertTrue(by[k]["hypothesis"].startswith("待驗證假設"))

    def test_daily_acceptance_answers_seven_questions_honestly(self):
        rep = report()
        evo = {"reflection": ev.reflect(rep), "new_experiments": [], "governance": {"label": "未通過", "recommend": False, "gates": [{"gate": "前瞻驗證", "ok": False}]}}
        A = ev.daily_acceptance(rep, evo, {"tasks": []})
        self.assertEqual(len(A), 7)
        self.assertFalse(A[0]["yes"])
        self.assertIn("沒有", A[0]["answer"])
        self.assertIn("還不知道", A[4]["answer"])
        self.assertIn("前瞻驗證", A[6]["answer"])
        self.assertTrue(all(a["evidence"] for a in A))

    def test_run_history_failure_kinds(self):
        tmp = Path(tempfile.mkdtemp()) / "runs.json"
        tmp.write_text(json.dumps({"runs": [{"at": "a", "ok": True}, {"at": "b", "ok": False, "error": "Actions 步驟結果：cancelled（可能逾時或環境錯誤）"},
                                            {"at": "c", "ok": False, "error": "ConnectionError: x"}]}), "utf-8")
        st = dl.run_status(tmp)
        self.assertEqual(len(st["history"]), 3)
        self.assertEqual(st["history"][0]["at"], "c")     # 最新在前
        self.assertEqual(st["fail_kinds"], {"逾時": 1, "網路或資料來源": 1})

    def test_partial_step_failures_are_alerted(self):
        """整體跑完但有步驟失敗（Actions 仍顯示成功），連續 2 次也要警示，不能藏起來。"""
        tmp = Path(tempfile.mkdtemp()) / "runs.json"
        with mock.patch.dict("os.environ", {"GITHUB_RUN_ID": "1"}):
            dl.record(True, None, "2026-10-20", 5, path=tmp)
        with mock.patch.dict("os.environ", {"GITHUB_RUN_ID": "2"}):
            dl.record(True, None, "2026-10-21", 5, path=tmp, failed_steps=["錯誤分析"])
        st = dl.run_status(tmp)
        self.assertEqual(st["partial_streak"], 1)
        self.assertFalse(st["partial_alert"])
        with mock.patch.dict("os.environ", {"GITHUB_RUN_ID": "3"}):
            dl.record(True, None, "2026-10-22", 5, path=tmp, failed_steps=["錯誤分析", "模型競賽"])
        st = dl.run_status(tmp)
        self.assertTrue(st["partial_alert"])
        self.assertTrue(st["needs_human"])
        self.assertEqual(st["last_valid_report"], "2026-10-20")
        self.assertEqual(st["fail_streak"], 0)

    def test_loss_attribution_and_regimes(self):
        P = fake_preds()
        dates = sorted(P["date"].unique())
        la = ps.loss_attribution(P, dates)
        self.assertIn("finding", la) if la["big_loss"] >= 20 else self.assertIn("note", la)
        r = ps._by_regime({"dates": dates[:4], "series": [1.0, -1.0, 2.0, 0.0]}, {dates[0]: "up_calm", dates[1]: "up_calm", dates[2]: "down_wild"})
        self.assertEqual(r["up_calm"]["periods"], 2)
        self.assertNotIn(dates[3], r)


class ScoreBands(unittest.TestCase):
    def test_bands_split_by_score_and_measure_excess_vs_bench(self):
        from app import factor_research as fr
        idx = [f"s{i}" for i in range(60)]
        score = pd.Series(np.linspace(10, 95, 60), index=idx)
        ret = pd.Series(np.linspace(-0.05, 0.10, 60), index=idx)
        b = fr._bands(score, ret, 0.01)
        self.assertEqual(len(b), len(fr.BANDS))
        self.assertEqual(sum(x["n"] for x in b if x), 60)
        self.assertLess(b[0]["ex"], b[-1]["ex"])            # 這組假資料分數越高、報酬越高
        s = fr.band_summary([b, b])
        self.assertEqual(s[-1]["band"], "90 以上")
        self.assertEqual(s[0]["periods"], 2)
        self.assertIsNone(fr._bands(score, ret, None))       # 沒有基準就不算

