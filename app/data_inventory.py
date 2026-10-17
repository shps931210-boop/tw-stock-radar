"""AI 訓練資料盤點（data/data_inventory.json）。

在改用機器學習模型、加入新聞因子之前，先回答「資料夠不夠」：
1. 有幾年每日行情？  2. 營收與財報有沒有實際公告時間？  3. 有沒有包含下市、停牌的股票？
4. 新聞有沒有原始發布時間與系統取得時間？  5. 過去的模型分數能不能用當時的資料重新產生？
每一題都從資料庫實際查出來，並給出「足夠／部分／不足」與原因；不靠人工填寫。
"""
from __future__ import annotations

import json
from datetime import date

from . import config, db

LEVEL = {"ok": "足夠", "partial": "部分可用", "gap": "不足"}


def _one(con, sql, args=()):
    try:
        return con.execute(sql, args).fetchone()
    except Exception:
        return None


def _years(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    return round((date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days / 365.25, 1)


def _backfill_txt(con) -> str:
    """夜間長期歷史回補的進度（app.update.update_history 寫入）。"""
    try:
        h = json.loads(db.get_meta(con, "history_backfill") or "null")
    except Exception:
        h = None
    if not h or not h.get("target"):
        return ""
    left = max((h.get("left_days") or {}).values(), default=0)
    fin = (_one(con, "SELECT COUNT(*) FROM fetch_log WHERE source='fm_financials_hist'") or [0])[0]
    return (f"夜間正在把股價往前補到 {h['target']}，還有約 {left} 個平日要抓；" if left else f"股價已補到目標起點 {h['target']}；") + \
        f"財報往前補齊的股票 {fin} 檔。"


def build(con) -> dict:
    q = []
    # 1. 每日行情
    p0, p1, nd = _one(con, "SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM prices") or (None, None, 0)
    ns = (_one(con, "SELECT COUNT(DISTINCT stock_id) FROM prices") or [0])[0]
    yrs = _years(p0, p1)
    q.append({"id": "prices", "q": "可以可靠取得多少年每日台股行情？",
              "level": "ok" if (yrs or 0) >= 5 else "partial" if (yrs or 0) >= 2 else "gap",
              "facts": {"起": p0, "迄": p1, "年數": yrs, "交易日": nd, "股票數": ns},
              "why": f"資料庫有 {p0}～{p1}（約 {yrs} 年、{nd} 個交易日）的上市櫃收盤行情。"
                     + ("機器學習模型通常希望至少涵蓋一次完整多空循環（5 年以上），目前的長度只夠做初步研究，樣本外期間會很短。" if (yrs or 0) < 5 else "")
                     + _backfill_txt(con)})
    # 2. 公告時間
    r0 = (_one(con, "SELECT MIN(ym) FROM revenue") or [None])[0]
    f0, nf = _one(con, "SELECT MIN(period), COUNT(DISTINCT stock_id) FROM financials") or (None, 0)
    q.append({"id": "announce", "q": "過去的營收、財報有沒有實際公告日期？", "level": "partial",
              "facts": {"月營收起": r0, "季報起": f0, "有季報的股票": nf, "實際公告時間": "沒有記錄"},
              "why": "沒有。資料來源（FinMind）只提供資料期別，沒有每家公司實際公告與更正的時間，所以回測用法定期限推估可用日"
                     "（月營收：次月 10 日後；季報：Q1 5/15、Q2 8/14、Q3 11/14、年報 3/31 之後）。"
                     "另外歷史財報是更正後的版本，不一定是當時看得到的數字。這能當近似研究，但不能宣稱完全沒有前視偏誤。"})
    # 3. 下市、停牌
    recent = (_one(con, "SELECT date FROM (SELECT DISTINCT date FROM prices ORDER BY date DESC LIMIT 20) ORDER BY date LIMIT 1") or [None])[0]
    gone = (_one(con, "SELECT COUNT(*) FROM (SELECT stock_id, MAX(date) d FROM prices GROUP BY stock_id) WHERE d < ?", (recent,)) or [0])[0] if recent else 0
    q.append({"id": "survivorship", "q": "是否包含曾經下市、停牌的股票？", "level": "ok" if gone else "partial",
              "facts": {"最近 20 個交易日沒有成交的股票": gone},
              "why": f"有。行情資料是逐日抓全市場，所以期間內下市或長期停牌的股票（目前 {gone} 檔最近 20 個交易日沒有成交）仍留在資料庫，回測不會只看活下來的股票。"
                     "但 {0} 以前已經下市的股票不在資料裡。".format(p0 or "資料起始日")})
    # 4. 新聞
    n0, n1, nn, nt, nfa = _one(con, "SELECT MIN(date), MAX(date), COUNT(*), SUM(time IS NOT NULL AND time != ''), SUM(fetched_at IS NOT NULL) FROM news_items") or (None, None, 0, 0, 0)
    nyrs = _years(n0, n1)
    q.append({"id": "news", "q": "過去新聞有沒有原始發布時間及完整內容？", "level": "gap" if (nyrs or 0) < 1 else "partial",
              "facts": {"新聞起": n0, "新聞迄": n1, "筆數": nn, "有發布時刻": nt or 0, "有系統取得時間": nfa or 0},
              "why": (f"新聞與公告目前保存了 {n0}～{n1}（{nn} 筆）。每則都有日期與系統第一次取得的時間，"
                      f"{'有' if nn and (nt or 0) / nn >= 0.8 else '只有 ' + str(round((nt or 0) / nn * 100)) + '% 有'}精確的發布時刻，" if nn else "目前資料庫沒有新聞。")
                     + "只保存標題與摘要、不是全文；財經新聞只涵蓋約 150 檔重點股票。以前只保留 90 天，從 2026-10-15 起改成長期保存。"
                     "新聞歷史太短，還不能做新聞因子的樣本外回測，要先持續累積至少一到兩年。"})
    # 5. 歷史分數能否重建
    pit = {}
    pp = config.ROOT / "history" / "backtest" / "ai_pit.json"
    if pp.exists():
        try:
            pit = json.loads(pp.read_text("utf-8"))
        except Exception:
            pit = {}
    months = len(pit.get("timeline") or [])
    picks = sorted((config.ROOT / "history" / "picks").glob("????-??-??.json"))
    q.append({"id": "pit", "q": "現有的歷史模型分數能否依當時資料重新產生？", "level": "partial" if months else "gap",
              "facts": {"重建起": pit.get("start"), "重建月數": months, "每日名單紀錄起": picks[0].stem if picks else None, "每日名單天數": len(picks)},
              "why": (f"可以近似重建：歷史回測從 {pit.get('start')} 起每月用當時可取得的股價與推估可用的財報重算五個因子（{months} 個月）。" if months else "還沒有歷史重建結果。")
                     + "但法人、融資、官方本益比只從 2026-08 起有資料，所以重建的模型不含籌碼；"
                     + (f"真正「當天發布、事後不改」的名單從 {picks[0].stem} 才開始保存。" if picks else "")})
    # 其他資料起點
    starts = {}
    for t, col, label in (("institutional", "date", "三大法人"), ("margin", "date", "融資融券"), ("valuation", "date", "官方本益比"),
                          ("exrights", "date", "除權息"), ("dividends", "year", "股利"), ("index_prices", "date", "加權指數")):
        r = _one(con, f"SELECT MIN({col}), MAX({col}) FROM {t}")
        starts[label] = {"起": r[0], "迄": r[1]} if r and r[0] else None
    worst = [x for x in q if x["level"] == "gap"]
    verdict = ("建議先做資料工程：" + "、".join(x["q"].rstrip("？") for x in worst) + "。這些缺口補上前，訓練更複雜的模型容易學到不存在的規律。") if worst else \
        "資料可以支援初步的模型研究，但結果只能當作近似驗證。"
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "questions": q,
            "levels": LEVEL, "starts": starts, "verdict": verdict}


def main() -> None:
    with db.connect() as con:
        print(json.dumps(build(con), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
