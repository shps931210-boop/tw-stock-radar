"""個股性質卡：用規則把每檔股票的數據翻成白話——它是什麼類型的股票、比較適合用什麼時間尺度觀察、財報重點。

全部由固定規則產生，不是預測，也不是買賣建議；每一句都附上依據的數字。
分數（score_*）是全市場百分位（0～100，50 = 中位數）。
"""
from __future__ import annotations

CYCLICAL = {"航運業", "鋼鐵工業", "塑膠工業", "水泥工業", "油電燃氣業", "玻璃陶瓷", "造紙工業", "化學工業"}


def _n(x):
    return None if x is None else float(x)


def _f(x, d=1):
    return "—" if x is None else f"{x:,.{d}f}"


def styles(s: dict) -> list[dict]:
    """股票類型（可能同時有好幾個）。"""
    q, g, v, m, t = (_n(s.get(f"score_{k}")) for k in ("quality", "growth", "value", "momentum", "trend"))
    out = []
    if q is not None and q >= 70 and (s.get("eps_ttm") or 0) > 0:
        why = f"體質分數 {q:.0f}（比全市場約 {min(q, 99):.0f}% 的股票好），ROE {_f(s.get('roe'))}%"
        if (s.get("roe_years_15") or 0) >= 3:
            why += f"，近年有 {s['roe_years_15']:.0f} 年 ROE 超過 15%"
        out.append({"label": "長期體質型", "why": why})
    if g is not None and g >= 70:
        out.append({"label": "成長型", "why": f"成長分數 {g:.0f}，近 3 個月營收年增 {_f(s.get('rev_yoy_3m'))}%"})
    if v is not None and v >= 70:
        out.append({"label": "價值型", "why": f"估值分數 {v:.0f}，本益比 {_f(s.get('per'))} 倍、股價淨值比 {_f(s.get('pbr'), 2)} 倍"})
    if (s.get("dy") or 0) >= 5 and (s.get("div_years") or 0) >= 5:
        out.append({"label": "高股息型", "why": f"殖利率 {_f(s.get('dy'))}%，連續 {s['div_years']:.0f} 年配息"})
    if m is not None and m >= 70 and (t or 0) >= 60:
        out.append({"label": "動能型", "why": f"動能分數 {m:.0f}、趨勢分數 {t:.0f}，近 6 個月漲 {_f(s.get('ret_6m'))}%"})
    if s.get("industry") in CYCLICAL:
        out.append({"label": "景氣循環型", "why": f"{s['industry']}的獲利跟著景氣、報價大起大落，財報好的時候常常已經接近高點"})
    if s.get("long_term_loss") or (s.get("eps_ttm") is not None and s["eps_ttm"] <= 0):
        out.append({"label": "虧損中", "why": f"近四季 EPS {_f(s.get('eps_ttm'), 2)} 元"})
    if (s.get("vol_1y_pct") or 0) >= 85:
        out.append({"label": "高波動", "why": f"年化波動率 {_f(s.get('vol_1y'))}%，比全市場約 {min(s['vol_1y_pct'], 99):.0f}% 的股票還大，漲跌幅度大"})
    return out


def horizon(s: dict) -> dict:
    """比較適合用什麼時間尺度觀察（規則判斷，不是買賣建議）。景氣循環股另外提醒。"""
    h = _horizon(s)
    if s.get("industry") in CYCLICAL and h["label"] not in ("資料不足", "只適合短線、風險高"):
        h["why"] += "但它是景氣循環股，長期持有要留意景氣反轉。"
    return h


def _horizon(s: dict) -> dict:
    q, m, t = (_n(s.get(f"score_{k}")) for k in ("quality", "momentum", "trend"))
    if not s.get("complete"):
        return {"label": "資料不足", "why": "必要資料還沒有到齊，先不判斷。"}
    losing = s.get("long_term_loss") or (s.get("eps_ttm") or 0) <= 0
    if losing:
        return {"label": "只適合短線、風險高", "why": "公司目前虧損，股價主要靠消息和資金推動，不適合長期放著。"}
    good = q is not None and q >= 60
    hot = m is not None and m >= 60 and (t or 0) >= 60
    if good and hot:
        return {"label": "長期、波段都可以觀察", "why": f"體質分數 {q:.0f} 不錯，股價趨勢也向上（動能 {m:.0f}）。"}
    if good:
        return {"label": "適合長期觀察", "why": f"體質分數 {q:.0f} 不錯，但股價目前沒有明顯趨勢，比較適合慢慢觀察、分批評估。"}
    if hot:
        return {"label": "比較偏波段", "why": f"股價趨勢向上（動能 {m:.0f}），但體質分數 {_f(q, 0)} 普通，漲勢不一定有基本面支撐。"}
    return {"label": "沒有明顯優勢", "why": "體質和股價趨勢都在普通水準，沒有特別適合的時間尺度。"}


def _lvl(x, good, weak, higher_better=True):
    if x is None:
        return "na"
    if higher_better:
        return "good" if x >= good else "weak" if x < weak else "mid"
    return "good" if x <= good else "weak" if x > weak else "mid"


def finance(s: dict) -> list[dict]:
    """財報重點：獲利、成長、現金流、財務結構、股利。level：good／mid／weak／na。"""
    fin = s.get("is_financial")
    out = []
    roe = _n(s.get("roe"))
    txt = f"ROE {_f(roe)}%" + ("" if fin else f"，毛利率 {_f(s.get('gross_margin'))}%、營業利益率 {_f(s.get('op_margin'))}%")
    out.append({"topic": "獲利能力", "level": _lvl(roe, 15, 8), "text": txt + {"good": "，賺錢效率好。", "mid": "，普通。", "weak": "，偏弱。", "na": "。"}[_lvl(roe, 15, 8)]})
    g = _n(s.get("rev_yoy_3m"))
    gt = f"近 3 個月營收年增 {_f(g)}%"
    if s.get("eps_ttm_turn"):
        gt += f"，近四季 EPS {s['eps_ttm_turn']}"
    elif s.get("eps_ttm_yoy") is not None:
        gt += f"，近四季 EPS 年增 {_f(s.get('eps_ttm_yoy'))}%"
    out.append({"topic": "成長", "level": _lvl(g, 10, 0), "text": gt + "。"})
    c = _n(s.get("ocf_to_ni"))
    lv = _lvl(c, 1, 0.5)
    out.append({"topic": "現金流", "level": lv, "text": f"營業現金流是淨利的 {_f(c, 2)} 倍"
                + {"good": "，獲利有收到現金。", "mid": "，大部分獲利有收到現金。", "weak": "，獲利有不少還沒收到現金，要留意。", "na": "。"}[lv]})
    d = _n(s.get("debt_ratio"))
    if fin:
        out.append({"topic": "財務結構", "level": "na", "text": f"負債比 {_f(d)}%。金融業本來就靠負債（存款）經營，負債比高是正常的。"})
    else:
        lv = _lvl(d, 40, 60, higher_better=False)
        out.append({"topic": "財務結構", "level": lv,
                    "text": f"負債比 {_f(d)}%" + {"good": "，財務穩健。", "mid": "，普通。", "weak": "，偏高，景氣不好時壓力較大。", "na": "。"}[lv]})
    dy = _n(s.get("dy"))
    out.append({"topic": "股利", "level": _lvl(dy, 4, 1.5),
                "text": f"殖利率 {_f(dy)}%，連續 {_f(s.get('div_years'), 0)} 年配息" + (f"，配息率 {_f(s.get('payout'), 0)}%。" if s.get("payout") is not None else "。")})
    return out


def build(s: dict) -> dict:
    return {"styles": styles(s), "horizon": horizon(s), "finance": finance(s)}


def annotate(rows: list[dict]) -> None:
    for r in rows:
        if r.get("is_common"):
            r["profile"] = build(r)
