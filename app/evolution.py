"""V15 AI 自主進化引擎：目標 → 記憶 → 反省 → 研究任務 → 受控實驗 → 晉升把關。

沿用既有架構，不另建一套：
- 預測記憶＝history/predictions/{日期}.json（ml_predict 封存，不可覆寫）
- 結果記憶＝history/learning/matured/{訊號日}_h{天數}.json（daily_learning.mature，只寫一次）
- 錯誤記憶＝history/learning/research_queue.json＋每日報告的 errors（只增不刪）
- 實驗記憶＝history/research/experiments.json：items 是人工整理的決定，auto 是每次研究自動登記的實驗（本模組）
- 目標記憶＝config/ai_mission.yaml（程式只讀）＋history/research/mission_log.json（版本變動紀錄）

這裡的「反省」「研究」都是規則與統計，不呼叫語言模型、不產生任何金融數字；每一句結論都附上來源檔案。
研究代理只能排程白名單內的實驗（lgbm_study、portfolio_study），不能改正式模型、預測紀錄或交易邏輯。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parent.parent
MISSION = ROOT / "config" / "ai_mission.yaml"
RESEARCH = ROOT / "history" / "research"
EXPERIMENTS = RESEARCH / "experiments.json"
MISSION_LOG = RESEARCH / "mission_log.json"
TASKS = RESEARCH / "research_tasks.json"
LEARN = ROOT / "history" / "learning"
BT = ROOT / "history" / "backtest"
RETRY_FAILS = 2          # 同一個實驗連續失敗幾次之後，要有足夠新資料才重跑
RETRY_NEW_PERIODS = 6    # 「足夠新資料」＝至少多 6 個不重疊的 20 日期間（約半年）
STUDIES = {  # 白名單：研究代理能排程的實驗，以及它們調整的東西
    "lgbm_study": {"file": "lgbm_study.json", "kinds": ("target", "train_window"), "goal": "G1"},
    "portfolio_study": {"file": "portfolio_study.json", "kinds": ("portfolio_rule",), "goal": "G1"},
}


def _read(p: Path, default=None):
    try:
        return json.loads(p.read_text("utf-8")) if p.exists() else default
    except Exception:  # noqa: BLE001
        return default


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), "utf-8")


def _now() -> str:
    return config.now().isoformat(timespec="seconds")


# ---------------- 1. 目標記憶 ----------------
def load_mission(path: Path | None = None) -> dict:
    import yaml
    path = path or MISSION
    raw = path.read_bytes()
    m = yaml.safe_load(raw)
    m["hash"] = hashlib.sha256(raw).hexdigest()[:12]
    return m


def goal_text(gid: str, mission: dict | None = None) -> str:
    m = mission or load_mission()
    g = next((g for g in m["goals"] if g["id"] == gid), None)
    return f"{gid}（{g['priority']}）{g['text']}" if g else gid


def track_mission(mission: dict | None = None, path: Path | None = None) -> dict:
    """每天記下目標檔的版本與雜湊；內容變了但版本沒加一，就標成「未經版本更新的修改」。"""
    m = mission or load_mission()
    path = path or MISSION_LOG
    log = _read(path, {"entries": []})
    last = log["entries"][-1] if log["entries"] else None
    status = "unchanged"
    if last is None:
        status = "first"
    elif last["hash"] != m["hash"]:
        status = "changed" if m["version"] > last["version"] else "changed_without_version"
    if status != "unchanged":
        log["entries"].append({"at": _now(), "version": m["version"], "hash": m["hash"], "status": status})
        _write(path, log)
    return {"version": m["version"], "hash": m["hash"], "status": status, "alert": status == "changed_without_version"}


# ---------------- 2. 實驗記憶 ----------------
def _exp_key(family: str, spec: str, data_end: str | None) -> str:
    return hashlib.sha256(f"{family}|{spec}|{data_end}".encode()).hexdigest()[:12]


def register(rec: dict, path: Path | None = None) -> tuple[dict, bool]:
    """登記一筆實驗。同一個實驗（同一家族、同一設定、同一個資料截止日）只記一次。回傳 (紀錄, 是否新增)。失敗的實驗也記、永遠不刪。"""
    path = path or EXPERIMENTS
    E = _read(path, {"schema_version": 1, "items": []})
    E.setdefault("auto", [])
    key = _exp_key(rec["family"], rec["spec"], rec.get("data_end"))
    old = next((x for x in E["auto"] if x["key"] == key), None)
    if old:
        return old, False
    same = [x for x in E["auto"] if x["family"] == rec["family"] and x["spec"] == rec["spec"]]
    n = sum(1 for x in E["auto"] if x["id"].startswith("EXP-" + rec["run_at"][:10].replace("-", ""))) + 1
    x = {"id": f"EXP-{rec['run_at'][:10].replace('-', '')}-{n:03d}", "key": key, **rec,
         "related": [y["id"] for y in same][-5:], "repeat_of_failed": sum(1 for y in same if not y.get("passed"))}
    E["auto"].append(x)
    _write(path, E)
    return x, True


def history_of(family: str, spec: str, path: Path | None = None) -> list[dict]:
    E = _read(path or EXPERIMENTS, {}) or {}
    return [x for x in E.get("auto", []) if x["family"] == family and x["spec"] == spec]


def ingest_studies(bt: Path | None = None, path: Path | None = None) -> list[dict]:
    """把每週研究（lgbm_study、portfolio_study）的每個變體登記成實驗。已登記過的不重複。"""
    bt = bt or BT
    added = []
    s = _read(bt / "lgbm_study.json")
    if s and s.get("results"):
        base = s["results"].get("base", {})
        for k, r in s["results"].items():
            if k in ("base", "v21"):
                continue
            vb = r.get("vs_base") or {}
            x, new = register({"run_at": str(s.get("generated_at") or _now()), "family": "lgbm_study", "spec": k, "kind": "target" if k.startswith("t_") or k.startswith("fuse") else "train_window",
                               "goal": "G1", "hypothesis": r.get("name"), "baseline": "base（LightGBM 排名目標、全部歷史）",
                               "model_version": s.get("study_version"), "data_end": s.get("last_test"), "periods": s.get("independent_periods"),
                               "oos": {"net_excess": (r.get("portfolio") or {}).get("net_excess"), "ic": (r.get("ic") or {}).get("ic"),
                                       "mdd": (r.get("portfolio") or {}).get("mdd"), "holdout": (r.get("holdout") or {}).get("net_excess")},
                               "vs_baseline": vb, "baseline_oos": {"net_excess": (base.get("portfolio") or {}).get("net_excess")},
                               "cost_pct": 0.585, "t_need": s.get("t_need"), "passed": bool((r.get("verdict") or {}).get("passed")),
                               "conclusion": (r.get("verdict") or {}).get("label"), "source": "history/backtest/lgbm_study.json"}, path)
            if new:
                added.append(x)
    p = _read(bt / "portfolio_study.json")
    if p and p.get("results"):
        for k, r in p["results"].items():
            if k in ("top10", "v21_top10"):
                continue
            x, new = register({"run_at": str(p.get("generated_at") or _now()), "family": "portfolio_study", "spec": k, "kind": "portfolio_rule", "goal": "G1",
                               "hypothesis": (p.get("names") or {}).get(k), "baseline": "前 10 名等權重", "model_version": p.get("study_version"),
                               "data_end": p.get("last"), "periods": r.get("periods"),
                               "oos": {k2: r.get(k2) for k2 in ("net_excess", "t", "mdd", "total", "turnover")}, "vs_baseline": r.get("vs_top10") or {},
                               "cost_pct": p.get("cost_pct"), "t_need": p.get("t_need"), "passed": bool((r.get("verdict") or {}).get("passed")),
                               "conclusion": (r.get("verdict") or {}).get("label") or r.get("note"), "source": "history/backtest/portfolio_study.json"}, path)
            if new:
                added.append(x)
    return added


def allowed(study: str, periods_now: int | None = None, path: Path | None = None) -> dict:
    """研究代理的把關：同一個研究的所有變體都已連續失敗 RETRY_FAILS 次，而且新資料不夠，就不重跑。"""
    E = _read(path or EXPERIMENTS, {}) or {}
    xs = [x for x in E.get("auto", []) if x["family"] == study]
    if not xs:
        return {"run": True, "why": "還沒做過"}
    runs = sorted({x["data_end"] for x in xs if x.get("data_end")})
    last = [x for x in xs if x.get("data_end") in runs[-RETRY_FAILS:]]
    if len(runs) >= RETRY_FAILS and last and not any(x.get("passed") for x in last):
        p_last = max((x.get("periods") or 0) for x in xs)
        if periods_now is not None and periods_now - p_last < RETRY_NEW_PERIODS:
            return {"run": False, "why": f"最近 {RETRY_FAILS} 次全部沒通過，新資料只多 {periods_now - p_last} 期（需要 {RETRY_NEW_PERIODS} 期）"}
    return {"run": True, "why": "有新資料或還有變體沒定論"}


# ---------------- 3. 反省 ----------------
def reflect(rep: dict, queue: dict | None = None, live: dict | None = None, new_experiments: list | None = None) -> dict:
    """每天回答六個問題。每一句都附來源；原因不能確定的，標成「待驗證假設」。"""
    err = rep.get("errors") or {}
    mt = rep.get("matured_today") or []
    today_sigs = {x["signal_date"] for x in mt if str(x.get("horizon")) == "20"}
    worst = [w for w in err.get("worst", []) if w.get("sig") in today_sigs]
    q = (queue or {}).get("items", [])
    recurring = sorted([x for x in q if x["times"] >= 3 and x["status"] == "待驗證"], key=lambda x: -x["times"])[:5]
    rows = (live or {}).get("rows", [])
    trend = _trend(rows)
    labels = sum((rep.get("candidate", {}).get("new_labels") or {}).values())
    matured = rep.get("matured_stocks_today", 0)
    outcome = [
        {"kind": "新增資料", "done": bool(matured or labels), "evidence": f"到期對帳 {matured} 筆、新訓練標籤 {labels} 筆"},
        {"kind": "完成分析", "done": bool(err.get("dates")), "evidence": f"錯誤分析涵蓋 {err.get('dates', 0)} 個已到期訊號日" if err.get("dates") else "沒有到期的 20 日預測可以分析"},
        {"kind": "完成實驗", "done": bool(new_experiments), "evidence": f"新登記 {len(new_experiments or [])} 個實驗" if new_experiments else "今天沒有新的實驗"},
        {"kind": "訓練候選模型", "done": bool(rep.get("candidate", {}).get("retrained")), "evidence": rep.get("candidate", {}).get("version") or rep.get("candidate", {}).get("reason", "")},
        {"kind": "驗證候選模型改善", "done": (rep.get("gate") or {}).get("status") == "passed", "evidence": (rep.get("gate") or {}).get("label") or "—"},
    ]
    supervised = bool(matured or labels)
    return {
        "q1_new_errors": {"items": worst[:5], "evidence": "history/learning/matured/" if worst else None,
                          "text": f"今天到期的預測裡，落後最多的 {len(worst[:5])} 檔" if worst else "今天沒有新到期的 20 日預測，沒有新的錯誤證據"},
        "q2_recurring": {"items": [{"text": x["text"], "times": x["times"], "first_seen": x["first_seen"]} for x in recurring],
                         "evidence": "history/learning/research_queue.json",
                         "text": "重複出現 3 次以上的問題" if recurring else "還沒有重複出現 3 次以上的問題"},
        "q3_degrading": trend,
        "q4_hypotheses": {"items": (rep.get("feature_effect") or {}).get("hypotheses", []), "status": "待驗證假設",
                          "text": "以下是根據統計提出的待驗證假設，不是已確認的原因"},
        "q5_outcome": outcome,
        "supervised_learning": supervised,
        "supervised_text": "今天有新的真實結果可以學習" if supervised else "今天沒有新的到期結果或訓練標籤：只做了資料品質與漂移監控，沒有新的監督式學習",
    }


def _trend(rows: list[dict]) -> dict:
    """真實前瞻帳本：最新和 20 天前比，Rank IC、前 10 名超額有沒有變差。樣本少時只說樣本不足。"""
    ok = [r for r in rows if (r.get("models") or {}).get("v21")]
    if len(ok) < 2 or (ok[-1].get("matured_dates") or 0) < 20:
        n = ok[-1].get("matured_dates") if ok else 0
        return {"status": "accumulating", "text": f"已到期的 20 日訊號日只有 {n or 0} 個（至少要 20 個才判斷變好或變差）", "evidence": "history/learning/live_ledger.json"}
    a, b = ok[max(0, len(ok) - 21)]["models"], ok[-1]["models"]
    out = []
    for m in b:
        if m in a:
            for k, lab in (("ic", "Rank IC"), ("top_net", "前 10 名扣成本超額")):
                if a[m].get(k) is not None and b[m].get(k) is not None and b[m][k] < a[m][k]:
                    out.append(f"{m} 的{lab}從 {a[m][k]} 降到 {b[m][k]}")
    return {"status": "warn" if out else "ok", "items": out, "text": "；".join(out) if out else "和 20 天前相比沒有變差",
            "evidence": "history/learning/live_ledger.json"}


# ---------------- 4. 研究代理：研究任務 ----------------
SEVERITY = {"live_drop": 3, "industry": 2, "feature": 2, "data": 3, "stock": 1, "near_miss": 2}


def build_tasks(rep: dict, queue: dict | None, path: Path | None = None, exp_path: Path | None = None, bt: Path | None = None) -> dict:
    """從錯誤記憶、真實前瞻、研究的「方向較好但證據不足」產生研究任務。只會新增或更新，不會刪除。"""
    path = path or TASKS
    T = _read(path, {"schema_version": 1, "note": "研究代理產生的研究任務。只增不刪；狀態由每週研究或人工更新", "tasks": []})
    by = {t["key"]: t for t in T["tasks"]}
    today = rep.get("date") or _now()[:10]
    wl = set(load_mission().get("research_whitelist", []))
    cand = []
    for x in (queue or {}).get("items", []):
        if x["status"] != "待驗證":
            continue
        kind = {"industry": "portfolio_rule", "feature": "feature", "data": "data", "stock": "data"}.get(x["kind"], "data")
        cand.append({"key": f"q:{x['kind']}:{x['key']}", "source": x["kind"], "times": x["times"], "question": x["text"], "action": kind,
                     "goal": {"industry": "G4", "feature": "G3", "data": "G6", "stock": "G4"}.get(x["kind"], "G1")})
    for fam, f in STUDIES.items():
        s = _read((bt or BT) / f["file"]) or {}
        for k, r in (s.get("results") or {}).items():
            vb = r.get("vs_base") or r.get("vs_top10") or {}
            if (r.get("verdict") or {}).get("passed") is False and (vb.get("diff") or 0) > 0:
                cand.append({"key": f"nm:{fam}:{k}", "source": "near_miss", "times": 1, "action": f["kinds"][0], "goal": f["goal"],
                             "question": f"{(s.get('names') or {}).get(k) or r.get('name') or k}：方向較好（每期 {vb.get('diff')}%，t {vb.get('t')}）但證據不足，要不要放進影子預測累積真實前瞻？"})
    tr = (rep.get("reflection") or {}).get("q3_degrading") or {}
    if tr.get("status") == "warn":
        cand.append({"key": f"live:{today[:7]}", "source": "live_drop", "times": 1, "action": "model_params", "goal": "G3",
                     "question": "真實前瞻成績變差：" + tr["text"]})
    for c in cand:
        prior = history_of(*c["key"].split(":")[1:3], path=exp_path) if c["source"] == "near_miss" else []
        fails = sum(1 for p in prior if not p.get("passed"))
        score = SEVERITY.get(c["source"], 1) * (1 + min(c["times"], 5) / 5) * (2 if c["goal"] in ("G1", "G2", "G3", "G4") else 1) / (1 + fails)
        t = by.get(c["key"])
        if t is None:
            t = {"id": f"RT-{today.replace('-', '')}-{len(T['tasks']) + 1:03d}", "key": c["key"], "created": today, "status": "待研究",
                 "question": c["question"], "goal": c["goal"], "goal_text": goal_text(c["goal"]),
                 "hypothesis": "待驗證假設：" + c["question"],
                 "metric": "20 日前 10 名扣成本超額報酬（主要）；Rank IC、跑贏 0050 比例、最大回撤（輔助）",
                 "data_needed": "已到期的封存預測（history/learning/matured）與研究資料集（ml_panel）",
                 "design": "和現行做法在同一組測試日、同一成本、同一母體比較",
                 "oos": "逐期 walk-forward（訓練標籤要在測試日前到期）；之後放進影子預測累積真實前瞻",
                 "budget": "GitHub Actions 免費額度內，每週回測工作最多 330 分鐘",
                 "stop_rule": f"連續 {RETRY_FAILS} 次沒有通過、新資料不足 {RETRY_NEW_PERIODS} 期就暫停",
                 "action": c["action"], "allowed": c["action"] in wl, "needs_human": c["action"] not in wl}
            T["tasks"].append(t)
            by[c["key"]] = t
        t.update(last_seen=today, priority=round(score, 2), prior_failed_experiments=fails, question=c["question"])
    T["updated"] = today
    open_ = sorted([t for t in T["tasks"] if t["status"] == "待研究"], key=lambda t: -t["priority"])
    T["top"] = [t["id"] for t in open_[:5]]
    _write(path, T)
    return T


# ---------------- 5. 晉升把關：五道關卡 ----------------
def governance(rep: dict, bt: Path | None = None) -> dict:
    """研究模型 → 候選模型 → 正式模型。五關全過也只是「建議升級」，要人工核准；不通過就維持現狀。"""
    from . import learning
    audit = _read((bt or BT) / "audit.json") or {}
    g = rep.get("gate") or {}
    crit = {c["name"]: c for c in (g.get("criteria") or [])}
    pick = lambda key: next((c for n, c in crit.items() if key in n), None)
    days, need = g.get("days") or 0, g.get("need") or learning.GATE["min_days"]
    gates = [
        {"gate": "資料正確", "ok": bool(audit) and audit.get("passed") == audit.get("total"), "value": f"資料稽核 {audit.get('passed', 0)}/{audit.get('total', 0)} 項通過" if audit else "沒有稽核結果"},
        {"gate": "排序改善", "ok": bool((pick("Rank IC") or {}).get("ok")), "value": (pick("Rank IC") or {}).get("value") or "樣本不足"},
        {"gate": "策略改善", "ok": bool((pick("超額報酬") or {}).get("ok")) and bool((pick("0050") or {}).get("ok")), "value": (pick("超額報酬") or {}).get("value") or "樣本不足"},
        {"gate": "風險合理", "ok": bool((pick("回撤") or {}).get("ok")), "value": (pick("回撤") or {}).get("value") or "樣本不足"},
        {"gate": "前瞻驗證", "ok": days >= need, "value": f"已到期的真實預測 {days}／{need} 天"},
    ]
    ok = all(x["ok"] for x in gates)
    return {"tiers": {"champion": "V21 固定權重（正式推薦）", "challenger": rep.get("challenger") or "LightGBM 每週重新訓練（影子預測）",
                      "research": "lgbm_study、portfolio_study 的變體（只做研究）"},
            "gates": gates, "recommend": ok, "human_approval": True,
            "label": "五關都通過：建議升級，等網站主人核准" if ok else f"未通過：{'、'.join(x['gate'] for x in gates if not x['ok'])}。正式模型維持 V21"}


def memory_status() -> dict:
    """五種記憶目前有多少紀錄、放在哪裡。"""
    E = _read(EXPERIMENTS, {}) or {}
    Q = _read(LEARN / "research_queue.json", {"items": []})
    preds = sorted((ROOT / "history" / "predictions").glob("????-??-??.json"))
    mat = sorted((LEARN / "matured").glob("*_h20.json")) if (LEARN / "matured").exists() else []
    m = load_mission()
    return {"prediction": {"n": len(preds), "last": preds[-1].stem if preds else None, "path": "history/predictions/", "rule": "開盤前封存，不可覆寫"},
            "outcome": {"n": len(mat), "last": mat[-1].stem.split("_")[0] if mat else None, "path": "history/learning/matured/", "rule": "到期才對帳，只寫一次"},
            "error": {"n": len(Q["items"]), "path": "history/learning/research_queue.json", "rule": "只增不刪"},
            "experiment": {"n": len(E.get("auto", [])) + len(E.get("items", [])), "auto": len(E.get("auto", [])), "decisions": len(E.get("items", [])),
                           "path": "history/research/experiments.json", "rule": "失敗也記，不刪除"},
            "mission": {"version": m["version"], "hash": m["hash"], "goals": len(m["goals"]), "path": "config/ai_mission.yaml", "rule": "程式只讀，人工改版"}}


def evolve(rep: dict, queue: dict | None = None, live: dict | None = None) -> dict:
    """每日學習最後一步：目標檢查、登記新實驗、反省、更新研究任務、晉升把關。任何一步失敗都不影響正式預測。"""
    out, steps = {}, []

    def step(name, fn):
        try:
            v = fn()
            steps.append({"step": name, "status": "success"})
            return v
        except Exception as e:  # noqa: BLE001
            steps.append({"step": name, "status": "failed", "why": f"{type(e).__name__}: {e}"[:200]})
            return None
    out["mission"] = step("目標檢查", track_mission)
    new = step("登記新實驗", ingest_studies) or []
    out["new_experiments"] = [{"id": x["id"], "family": x["family"], "spec": x["spec"], "passed": x["passed"]} for x in new]
    out["reflection"] = step("反省", lambda: reflect(rep, queue, live, new))
    rep2 = {**rep, "reflection": out["reflection"] or {}}
    T = step("更新研究任務", lambda: build_tasks(rep2, queue))
    out["tasks"] = {"open": sum(1 for t in (T or {}).get("tasks", []) if t["status"] == "待研究"), "top": [t for t in (T or {}).get("tasks", []) if t["id"] in (T or {}).get("top", [])]}
    out["governance"] = step("晉升把關", lambda: governance(rep))
    out["memory"] = step("記憶盤點", memory_status)
    out["steps"] = steps
    return out


def main(argv: list[str]) -> int:
    if argv[:1] == ["allowed"] and len(argv) > 1:   # 工作流程用：python -m app.evolution allowed lgbm_study
        a = allowed(argv[1])
        print(("執行" if a["run"] else "跳過") + "：" + a["why"])
        return 0 if a["run"] else 3
    if argv[:1] == ["ingest"]:
        print(f"新登記 {len(ingest_studies())} 個實驗")
        return 0
    print(json.dumps(memory_status(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
