"""公告後股價反應：通過事前登記門檻的規則（R2 月營收、Q1 季報）只做「影子紀錄」（2026-11-20）。

- 訊號日收盤後封存名單到 history/predictions/announce/{訊號日}.json，已經存在就不覆寫（不得事後修改預測）。
- 到期後用同一套成交規則（隔天開盤買、到期日開盤賣、扣成本）對帳 0050，寫 history/predictions/announce/results.json。
- 不影響正式選股或買賣計畫；要升級必須有足夠的前瞻結果，並經使用者同意。

用法：python -m app.announce_shadow
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import announce_study as A
from . import config, db
from . import pit_backtest as pb

DIR = config.ROOT / "history" / "predictions" / "announce"
RULES = {"R2": A.NAMES["R2"], "Q1": A.NAMES["Q1"]}
MIN_SEALED_TO_JUDGE = 5


def _read(p: Path):
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def trading_days(dbname: str = "stocks.db", since: str = "2026-01-01") -> list[str]:
    with db.connect(dbname) as con:
        return [r[0] for r in con.execute("SELECT date FROM prices WHERE stock_id='0050' AND date>=? ORDER BY date", (since,))]


def due_today(days: list[str]) -> dict:
    """最後一個交易日是不是訊號日：R2 ＝ 當月第一個 ≥ 11 日的交易日；Q1 ＝ 第一個 ≥ 季報法定期限隔天的交易日。"""
    if len(days) < 2:
        return {}
    d, prev = days[-1], days[-2]
    out = {}
    eleven = d[:8] + "11"
    if d >= eleven and prev < eleven:
        p = pd.Period(d[:7], "M") - 1
        out["R2"] = str(p)
    y = int(d[:4])
    for per in (f"{y - 1}-Q4", f"{y}-Q1", f"{y}-Q2", f"{y}-Q3"):
        av = pb.fin_available(per)
        if d >= av > prev:
            out["Q1"] = per
    return out


def _exit_index(dates: list[str], rule: str, i_sig: int) -> int | None:
    if rule == "Q1":
        j = i_sig + 1 + A.HOLD
    else:
        nm = pd.Period(dates[i_sig][:7], "M") + 1
        k = A._first_on_or_after(dates, f"{nm.year}-{nm.month:02d}-11")
        j = None if k is None else k + 1
    return j if j is not None and j < len(dates) else None


def seal(D: dict, due: dict, out: Path = DIR) -> list[str]:
    dates, i = D["dates"], len(D["dates"]) - 1
    sig = dates[i]
    f = out / f"{sig}.json"
    if f.exists():
        return []
    rec = {"schema_version": 1, "signal_date": sig, "generated_at": config.now().isoformat(timespec="seconds"),
           "status": "shadow", "prereg": "docs/prereg_announce_1120.md", "study": "history/backtest/announce.json",
           "trade": "訊號日收盤後封存，隔一個交易日開盤買，等權 10 檔；R2 持有到下個月訊號日的隔天開盤，Q1 持有 20 個交易日；扣成本＋滑價 0.2%",
           "rules": {}}
    if "R2" in due:
        P = A.monthly_picks(D, A.surge_table(D["rev"]), pd.Period(due["R2"], "M"), i)
        rec["rules"]["R2"] = {"name": RULES["R2"], "ym": due["R2"], "picks": P["R2"], "universe": P["universe"]}
    if "Q1" in due:
        P = A.quarterly_picks(D, A.sue_table(D["fin"]), due["Q1"], i)
        rec["rules"]["Q1"] = {"name": RULES["Q1"], "period": due["Q1"], "picks": P["Q1"], "universe": P["universe"]}
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), "utf-8")
    return list(rec["rules"])


def score(D: dict, out: Path = DIR) -> dict:
    dates, rows = D["dates"], []
    for f in sorted(out.glob("????-??-??.json")):
        rec = _read(f) or {}
        sig = rec.get("signal_date")
        if sig not in dates:
            continue
        i = dates.index(sig)
        for rule, r in (rec.get("rules") or {}).items():
            j = _exit_index(dates, rule, i)
            row = {"signal": sig, "rule": rule, "entry": dates[i + 1] if i + 1 < len(dates) else None,
                   "exit": dates[j] if j is not None else None, "matured": j is not None}
            if j is not None:
                row |= {k: round(v * 100, 2) if isinstance(v, float) and k != "n" else v for k, v in A.hold(D, r["picks"], i + 1, j).items()}
            rows.append(row)
    summ = {}
    for rule in RULES:
        m = [r for r in rows if r["rule"] == rule and r.get("excess") is not None]
        ex = np.array([r["excess"] for r in m])
        summ[rule] = {"name": RULES[rule], "sealed": sum(r["rule"] == rule for r in rows), "matured": len(m),
                      "mean_excess": round(float(ex.mean()), 2) if len(m) else None,
                      "beat_share": round(float((ex > 0).mean() * 100), 1) if len(m) else None,
                      "enough": len(m) >= MIN_SEALED_TO_JUDGE}
    res = {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "summary": summ, "rows": rows,
           "note": "影子紀錄，只觀察、不影響正式選股；至少 5 批到期才判斷，升級要使用者同意。"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    return res


def main(dbname: str = "stocks.db") -> dict:
    days = trading_days(dbname)
    due = due_today(days)
    have = list(DIR.glob("????-??-??.json")) if DIR.exists() else []
    if not due and not have:
        return {"sealed": [], "note": "今天不是訊號日，也還沒有封存紀錄"}
    D = A.load(dbname)
    sealed = seal(D, due) if due and D["dates"][-1] == days[-1] else []
    res = score(D)
    return {"due": due, "sealed": sealed, "summary": res["summary"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=1))
