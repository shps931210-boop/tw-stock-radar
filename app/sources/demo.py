"""示範資料產生器：建立一個內容完整但「全部是模擬數字」的資料庫（data/demo.db）。

用途是在沒有網路或還沒申請 token 時，先看網站長什麼樣子、測試策略邏輯。
公司名稱是真的，但股價、財報、法人、新聞全是亂數產生，不能拿來做任何判斷。
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import db
from .. import config

STOCKS = [
    ("2330", "台積電", "半導體業"), ("2317", "鴻海", "其他電子業"), ("2454", "聯發科", "半導體業"),
    ("2308", "台達電", "電子零組件業"), ("2382", "廣達", "電腦及週邊設備業"), ("2412", "中華電", "通信網路業"),
    ("2881", "富邦金", "金融保險業"), ("2882", "國泰金", "金融保險業"), ("2891", "中信金", "金融保險業"),
    ("2303", "聯電", "半導體業"), ("3711", "日月光投控", "半導體業"), ("2886", "兆豐金", "金融保險業"),
    ("1216", "統一", "食品工業"), ("2002", "中鋼", "鋼鐵工業"), ("2357", "華碩", "電腦及週邊設備業"),
    ("3008", "大立光", "光電業"), ("2345", "智邦", "通信網路業"), ("3034", "聯詠", "半導體業"),
    ("2379", "瑞昱", "半導體業"), ("6669", "緯穎", "電腦及週邊設備業"), ("3037", "欣興", "電子零組件業"),
    ("2408", "南亞科", "半導體業"), ("2603", "長榮", "航運業"), ("2609", "陽明", "航運業"),
    ("1301", "台塑", "塑膠工業"), ("1303", "南亞", "塑膠工業"), ("2207", "和泰車", "汽車工業"),
    ("5871", "中租-KY", "其他業"), ("2395", "研華", "電腦及週邊設備業"), ("1101", "台泥", "水泥工業"),
    ("2912", "統一超", "貿易百貨業"), ("4904", "遠傳", "通信網路業"), ("2105", "正新", "橡膠工業"),
    ("9910", "豐泰", "其他業"), ("1590", "亞德客-KY", "電機機械"), ("2059", "川湖", "其他電子業"),
    ("3017", "奇鋐", "電腦及週邊設備業"), ("2376", "技嘉", "電腦及週邊設備業"), ("8046", "南電", "電子零組件業"),
    ("6505", "台塑化", "油電燃氣業"), ("5483", "中美晶", "半導體業"), ("6488", "環球晶", "半導體業"),
    ("3293", "鈊象", "文化創意業"), ("8299", "群聯", "半導體業"), ("6147", "頎邦", "半導體業"),
]
OTC = {"5483", "6488", "3293", "8299", "6147"}
NEWS = {1: ["{n}月營收創同期新高", "{n}接獲大單 營運動能看旺", "外資調高{n}目標價", "{n}擴產搶攻AI商機"],
        0: ["{n}股東會今召開", "{n}董事會通過配息案", "{n}參加國際展會"],
        -1: ["{n}營收年減 需求疲軟", "外資下修{n}評等", "{n}毛利率下滑 獲利衰退"]}


def _rng(sid: str, salt: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.md5(f"{sid}-{salt}".encode()).hexdigest()[:8], 16))


def build(path_name: str = "demo.db", days: int = 520) -> str:
    p = db.path(path_name)
    if p.exists():
        p.unlink()
    end = pd.Timestamp(config.today()) - pd.offsets.BDay(0 if config.today().weekday() < 5 else 1)
    tdays = pd.bdate_range(end=end, periods=days)
    ds = [d.strftime("%Y-%m-%d") for d in tdays]
    mkt = np.cumsum(_rng("mkt", "x").normal(0.0005, 0.011, days))

    with db.connect(path_name) as con:
        db.upsert(con, "index_prices", [{"date": d, "close": round(20000 * np.exp(m), 2)} for d, m in zip(ds, mkt)])
        for sid, name, ind in STOCKS:
            r = _rng(sid, "p")
            fin = ind == "金融保險業"
            quality, growth, trend = r.uniform(0, 1), r.uniform(-0.6, 1), r.uniform(-1, 1)
            beta = r.uniform(0.5, 1.5)
            db.upsert(con, "stocks", [{"stock_id": sid, "name": name, "industry": ind,
                                       "market": "tpex" if sid in OTC else "twse"}])
            # ---- 財報（12 季）與月營收（36 月）----
            months = pd.period_range(end=tdays[-1].to_period("M") - 1, periods=36, freq="M")
            base_rev = r.uniform(3e9, 2e11) / 3
            mg = 0.012 * growth + r.normal(0, 0.04, 36)
            if growth > 0.6:
                mg[-4:] += np.linspace(0.02, 0.08, 4)  # 營收加速
            rev_m = base_rev * np.exp(np.cumsum(mg)) * (1 + 0.08 * np.sin(np.arange(36) / 12 * 2 * np.pi))
            db.upsert(con, "revenue", [{"stock_id": sid, "ym": str(m), "revenue": float(v)} for m, v in zip(months, rev_m)])
            gm = 0 if fin else r.uniform(0.08, 0.6) * (0.6 + 0.6 * quality)
            opm = r.uniform(0.05, 0.4) if fin else gm * r.uniform(0.25, 0.7)
            shares = r.uniform(1e9, 2.5e10)
            assets = base_rev * 12 * r.uniform(1.0, 2.5) * (8 if fin else 1)
            debt = r.uniform(0.85, 0.93) if fin else r.uniform(0.2, 0.7) * (1.2 - 0.5 * quality)
            q_end = pd.period_range(end=months[-1].asfreq("Q") - (0 if months[-1].month % 3 == 0 else 1), periods=12, freq="Q")
            fin_rows, ni_hist = [], []
            for i, q in enumerate(q_end):
                qm = [str(m) for m in q.asfreq("M", "e") - np.arange(3)[::-1]]
                qrev = float(sum(v for m, v in zip(months, rev_m) if str(m) in qm)) or base_rev * 3
                op = qrev * opm * (1 + r.normal(0, 0.15 * (1.2 - quality)))
                ni = op * r.uniform(0.75, 0.9)
                ni_hist.append(ni)
                a = assets * (1 + 0.02 * i)
                eq = a * (1 - debt)
                ocf = ni * r.uniform(0.6, 1.5) * (1 + 0.3 * quality)
                fin_rows.append({"stock_id": sid, "period": f"{q.year}-Q{q.quarter}",
                                 "period_end": q.end_time.strftime("%Y-%m-%d"), "revenue": qrev,
                                 "gross_profit": None if fin else qrev * gm, "operating_income": op, "net_income": ni,
                                 "eps": round(ni / shares, 2), "total_assets": a, "total_liabilities": a * debt,
                                 "equity": eq, "current_assets": None if fin else a * r.uniform(0.3, 0.6),
                                 "current_liabilities": None if fin else a * r.uniform(0.15, 0.4),
                                 "share_capital": shares * 10, "ocf": ocf, "capex": abs(ocf) * r.uniform(0.1, 0.8)})
            db.upsert(con, "financials", fin_rows)
            eps_ttm = sum(x["eps"] for x in fin_rows[-4:])
            bvps = fin_rows[-1]["equity"] / shares
            # ---- 股利 ----
            payout = r.uniform(0.3, 0.9)
            first = int(r.integers(2014, 2022))
            divs = [{"stock_id": sid, "year": y, "cash": round(max(eps_ttm, 0.3) * payout * (0.85 + 0.03 * (y - 2018)), 2),
                     "stock": 0.0} for y in range(first, tdays[-1].year)]
            db.upsert(con, "dividends", divs)
            # ---- 股價：用財報推估合理價位，加上大盤 beta 與個股趨勢 ----
            pe = r.uniform(8, 30)
            target = max(eps_ttm, 0.5) * pe
            idio = r.normal(0.0004 * trend, 0.017, days)
            if trend > 0.5:
                idio[-60:] += 0.002
                idio[-3:] += 0.012  # 製造突破
            lr = beta * np.diff(np.r_[0, mkt]) + idio
            close = np.exp(np.cumsum(lr))
            close = close / close[-1] * target
            o = close * (1 + r.normal(0, 0.006, days))
            h = np.maximum(o, close) * (1 + np.abs(r.normal(0, 0.008, days)))
            lo = np.minimum(o, close) * (1 - np.abs(r.normal(0, 0.008, days)))
            liq = r.uniform(2e7, 5e9) if sid != "3293" else 8e6
            vol = (liq / target) * r.lognormal(0, 0.35, days)
            vol[-3:] *= 1 + 1.5 * (trend > 0.5)
            rnd = lambda x: float(np.round(x, 2 if target < 100 else 1))
            db.upsert(con, "prices", [{"stock_id": sid, "date": d, "open": rnd(a), "high": rnd(b), "low": rnd(c),
                                       "close": rnd(e), "volume": float(int(v)), "value": float(v * e)}
                                      for d, a, b, c, e, v in zip(ds, o, h, lo, close, vol)])
            last_div = divs[-1]["cash"] if divs else 0
            db.upsert(con, "valuation", [{"stock_id": sid, "date": d, "per": round(c / eps_ttm, 2) if eps_ttm > 0 else None,
                                          "pbr": round(c / bvps, 2), "dy": round(last_div / c * 100, 2)}
                                         for d, c in zip(ds[-260:], close[-260:])])
            # ---- 法人、融資（近 60 日）----
            flow = r.normal(0.25 * trend, 1, 60) * vol[-60:] * 0.08
            db.upsert(con, "institutional", [{"stock_id": sid, "date": d, "foreign_net": float(int(f)),
                                              "trust_net": float(int(f * r.uniform(0, 0.4) + r.normal(0.1 * trend, 0.3) * v * 0.02)),
                                              "dealer_net": float(int(r.normal(0, 0.01) * v))}
                                             for d, f, v in zip(ds[-60:], flow, vol[-60:])])
            mb = 2e4 * np.exp(np.cumsum(r.normal(-0.002 * trend, 0.01, 60)))
            db.upsert(con, "margin", [{"stock_id": sid, "date": d, "margin_bal": float(int(m)), "short_bal": float(int(m * 0.05))}
                                      for d, m in zip(ds[-60:], mb)])
            # ---- 新聞 ----
            news = []
            for k in range(int(r.integers(2, 6))):
                s = 1 if r.random() < 0.4 + 0.3 * growth else (0 if r.random() < 0.6 else -1)
                pool = NEWS[s]
                news.append({"stock_id": sid, "date": (config.today() - timedelta(days=int(r.integers(0, 14)))).isoformat(),
                             "title": pool[int(r.integers(len(pool)))].format(n=name), "source": "示範新聞", "link": ""})
            db.upsert(con, "news", news)
        db.upsert(con, "disposition", [{"stock_id": "3293", "period": "示範用", "reason": "示範：處置股票", "updated": ""}])
        from datetime import datetime
        now = config.now().isoformat(timespec="seconds")
        for k in ("prices", "institutional", "margin", "valuation", "revenue", "financials", "dividends"):
            db.set_meta(con, f"updated_{k}", now)
        db.set_meta(con, "last_update", now)
        db.set_meta(con, "source", "demo")
    return str(p)
