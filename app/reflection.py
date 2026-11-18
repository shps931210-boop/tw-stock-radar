"""每日反省（2026-11-18）：每天自動檢查「為什麼輸 0050」→ history/learning/reflection.json（每天另存一份，寫過不改）。

使用者規則（11/17）：可以虧，但不能輸 0050。所以這裡只問一件事：每一種正式模式有沒有輸 0050？輸在哪裡？

只用已到期、合格的真實預測（history/audit/fair_compare.json，開盤到開盤、扣成本），不重新計算、不看回測。
每一條結論都附數字和樣本數；樣本不夠就寫「樣本不夠，先觀察」，不下結論。

拆解方法
- 跟 0050 同漲同跌的程度（beta）：每批「選股淨報酬」對「0050 報酬」做回歸。
- 扣掉大盤影響後的選股能力（alpha）：回歸截距。beta 小於 1 的模式在大跌時自然少跌，這不是選股能力。
- 大盤漲的批次、跌的批次分開看相對 0050 的差距。
- 最近 10 批 vs 之前，有沒有變好。
- 研究流程：做了多少測試、累積了多少真實到期證據（測試太多、證據太少就是在碰運氣）。

用法：python -m app.reflection
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import config

ROOT = Path(__file__).resolve().parent.parent
FAIR = ROOT / "history" / "audit" / "fair_compare.json"
EXP = ROOT / "history" / "research" / "experiments.json"
LESSONS = ROOT / "history" / "research" / "lessons.json"
OUT = ROOT / "history" / "learning" / "reflection.json"
ARCHIVE = ROOT / "history" / "learning" / "reflection"
MODES = {"stable": "穩健型", "balanced": "均衡型", "aggressive": "積極型"}
MIN_N = 8          # 至少 8 批才下結論
RECENT = 10        # 最近幾批
LOW_BETA, HIGH_BETA = 0.85, 1.2
ALPHA_BAD = -0.5   # 扣掉大盤影響後每批落後超過 0.5 個百分點
WORSE = 0.5        # 最近 10 批比之前差超過 0.5 個百分點


def _batches(fc: dict, h: int) -> list[dict]:
    B = [b for b in fc.get("batches", []) if b.get("eligible") and b.get("horizon") == h]
    return sorted(B, key=lambda b: b["signal_date"])


def mode_stats(B: list[dict], mode: str) -> dict | None:
    rows = [(b["signal_date"], (b["modes"].get(mode) or {}).get("open") or {}) for b in B]
    rows = [(d, x["net"], x["e0050"]) for d, x in rows if x.get("net") is not None and x.get("e0050") is not None]
    if not rows:
        return None
    d, y, x = zip(*rows)
    y, x = np.asarray(y, float), np.asarray(x, float)
    e = y - x
    n = len(e)
    out = {"n": n, "first": d[0], "last": d[-1], "mean_excess": round(float(e.mean()), 2),
           "beat_share": round(float((e > 0).mean()) * 100, 1), "lose_streak": 0}
    for v in e[::-1]:
        if v >= 0:
            break
        out["lose_streak"] += 1
    if n >= 3 and x.std() > 0:
        beta, alpha = np.polyfit(x, y, 1)
        out["beta"], out["alpha"] = round(float(beta), 2), round(float(alpha), 2)
    up, down = x > 0, x <= 0
    out["up"] = {"n": int(up.sum()), "excess": round(float(e[up].mean()), 2) if up.any() else None}
    out["down"] = {"n": int(down.sum()), "excess": round(float(e[down].mean()), 2) if down.any() else None}
    if n > RECENT:
        out["recent"] = round(float(e[-RECENT:].mean()), 2)
        out["before"] = round(float(e[:-RECENT].mean()), 2)
    return out


def findings(name: str, s: dict | None, s20: dict | None) -> list[dict]:
    """把數字翻成白話結論。level：bad（違反不能輸 0050）、warn（找到原因）、info（觀察）。"""
    if not s or s["n"] < MIN_N:
        return [{"level": "info", "code": "few", "text": f"{name}：到期批次只有 {0 if not s else s['n']} 批，樣本不夠，先觀察。"}]
    F = []
    rec = s.get("recent", s["mean_excess"])
    if rec < 0:
        F.append({"level": "bad", "code": "lose_0050",
                  "text": f"{name}最近 {min(RECENT, s['n'])} 批平均輸 0050 {abs(rec):.2f} 個百分點（違反「不能輸 0050」）。"})
    b, a = s.get("beta"), s.get("alpha")
    up, dn = s["up"]["excess"], s["down"]["excess"]
    if b is not None and b < LOW_BETA and up is not None and dn is not None and up < 0 < dn:
        F.append({"level": "warn", "code": "low_beta",
                  "text": f"{name}跟大盤同漲同跌的程度只有 0050 的 {b:.2f} 倍：大盤跌時贏 {dn:.2f}、大盤漲時輸 {abs(up):.2f}。"
                          "贏 0050 主要是跌得少，不是選得準。"})
    if b is not None and b > HIGH_BETA:
        F.append({"level": "warn", "code": "high_beta",
                  "text": f"{name}漲跌幅度是 0050 的 {b:.2f} 倍，等於放大槓桿：大盤一跌就輸很多。"})
    if a is not None and a < ALPHA_BAD:
        F.append({"level": "warn", "code": "bad_alpha",
                  "text": f"{name}扣掉大盤影響後，每批平均還差 {abs(a):.2f} 個百分點：這是真的選錯股票。"})
    if "recent" in s and s["recent"] < s["before"] - WORSE:
        F.append({"level": "warn", "code": "worse",
                  "text": f"{name}最近 {RECENT} 批（{s['recent']:+.2f}）比之前（{s['before']:+.2f}）差，沒有進步。"})
    if s["lose_streak"] >= 3:
        F.append({"level": "warn", "code": "streak", "text": f"{name}已經連續 {s['lose_streak']} 批輸 0050。"})
    if s20 and s20["n"] >= 3 and s20["mean_excess"] < 0:
        F.append({"level": "bad", "code": "lose_0050_20d",
                  "text": f"{name} 20 日到期 {s20['n']} 批平均輸 0050 {abs(s20['mean_excess']):.2f} 個百分點。"})
    if not F:
        F.append({"level": "info", "code": "ok", "text": f"{name}最近沒有輸 0050，也沒有找到明顯的問題（樣本仍少，不代表長期有效）。"})
    return F


def process_check(exp: dict, fc: dict) -> dict:
    """研究流程健康度：測試次數 vs 真實到期證據。"""
    items = exp.get("items") or []
    auto = exp.get("auto") or []
    dec = [x.get("decision") for x in items]
    tests = len(items) + len(auto)
    evid20 = len(_batches(fc, 20))
    since = 0
    for x in sorted(items, key=lambda x: x.get("date") or "")[::-1]:
        if x.get("decision") == "adopted":
            break
        since += 1
    F = []
    if tests >= 10 * max(evid20, 1):
        F.append({"level": "warn", "code": "overtest",
                  "text": f"已經做了 {tests} 個研究測試，但 20 日真實到期證據只有 {evid20} 批。"
                          "測試越多，越容易碰巧找到「看起來有效」的規則；現在該做的是累積真實證據，不是再加新候選。"})
    if since >= 5:
        F.append({"level": "warn", "code": "no_adopt",
                  "text": f"最近 {since} 個研究決定都沒有採用：在同一個框架裡換參數已經沒有用，要換問題，不是換參數。"})
    return {"tests": tests, "rejected": dec.count("rejected"), "shadow": dec.count("shadow"), "adopted": dec.count("adopted"),
            "evidence_20d": evid20, "since_last_adopted": since, "findings": F}


NEXT = {
    "lose_0050": "研究讓這個模式不輸 0050 的做法（例如以 0050 為主、選股為輔）；正式要改需要你同意。",
    "low_beta": "比較時要扣掉大盤影響：大跌時「贏」不算進步。",
    "high_beta": "限制高波動股票的比重（積極型波動上限影子紀錄已在累積）。",
    "bad_alpha": "這個模式的排序在真實預測裡沒有效，先不要再加新因子，等更多到期證據確認。",
    "worse": "檢查最近市場是不是換了風格（例如權值股帶動），選股是否沒有跟上。",
    "streak": "連續輸的批次逐檔看歸因（history/learning/attribution/）。",
    "lose_0050_20d": "20 日是主要指標：再累積到期批次，若持續輸就建議降級為研究用。",
    "overtest": "暫停新增候選規則，改成只追蹤已登記的影子紀錄到期結果。",
    "no_adopt": "下一個研究要換問題（例如以 0050 為比較基準），不是在原本的分數上調參數。",
}


def build(fc: dict | None = None, exp: dict | None = None, lessons: list | None = None) -> dict:
    fc = fc if fc is not None else (json.loads(FAIR.read_text("utf-8")) if FAIR.exists() else {})
    exp = exp if exp is not None else (json.loads(EXP.read_text("utf-8")) if EXP.exists() else {})
    lessons = lessons if lessons is not None else (json.loads(LESSONS.read_text("utf-8")).get("lessons", []) if LESSONS.exists() else [])
    B5, B20 = _batches(fc, 5), _batches(fc, 20)
    modes = {}
    for m, name in MODES.items():
        s, s20 = mode_stats(B5, m), mode_stats(B20, m)
        modes[m] = {"name": name, "h5": s, "h20": s20, "findings": findings(name, s, s20)}
    proc = process_check(exp, fc)
    codes = {f["code"] for v in modes.values() for f in v["findings"]} | {f["code"] for f in proc["findings"]}
    bad = [v["name"] for v in modes.values() if any(f["level"] == "bad" for f in v["findings"])]
    head = (f"{'、'.join(bad)}目前輸 0050，還沒有達到「可以虧但不能輸 0050」。" if bad
            else "目前三種模式都沒有輸 0050（樣本仍少）。" if B5 else "還沒有到期的真實預測可以反省。")
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"),
            "source": "history/audit/fair_compare.json（開盤到開盤、扣成本、只算合格批次）",
            "fair_generated_at": fc.get("generated_at"), "batches": {"h5": len(B5), "h20": len(B20)},
            "headline": head, "modes": modes, "process": proc,
            "next": [NEXT[c] for c in NEXT if c in codes], "lessons": lessons,
            "rule": "可以虧，但不能輸 0050（使用者 2026-11-17）。反省只陳述到期數字，改正式選股一律要你同意。"}


def run(out: Path = OUT, archive: Path = ARCHIVE) -> dict:
    r = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    archive.mkdir(parents=True, exist_ok=True)
    day = archive / f"{r['generated_at'][:10]}.json"
    if not day.exists():   # 每天第一次的反省保留下來，之後不改，可以看出反省內容怎麼變
        day.write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    return r


def main(argv: list[str]) -> int:
    r = run()
    print(r["headline"])
    for v in r["modes"].values():
        for f in v["findings"]:
            print(f"[{f['level']}] {f['text']}")
    for f in r["process"]["findings"]:
        print(f"[{f['level']}] {f['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
