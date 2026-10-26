"""V20 個股走勢分頁：策略引擎提供逐項條件與趨勢判斷；頁面順序是狀態 → 三期間 → 均線圖 → 條件 → 進出場，研究細節收合在後。"""
import re
import unittest

import numpy as np
import pandas as pd

from app import strategy

HTML = open("static/index.html", encoding="utf-8").read()


def px(trend=1.0, n=300):
    d = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d")
    c = 100 + np.arange(n) * trend + np.sin(np.arange(n) / 3) * 2
    return pd.DataFrame({"date": d, "open": c - 0.5, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6, "value": 5e7})


class Engine(unittest.TestCase):
    def test_checks_and_trend(self):
        st = strategy.status(px(1.0))
        self.assertEqual(st["trend"]["label"], "偏多")
        for m in ("short", "swing"):
            ks = [c["k"] for c in st[m]["checks"]]
            self.assertTrue(all(isinstance(c["ok"], bool) for c in st[m]["checks"]))
            self.assertTrue(any("成交量足夠" in k for k in ks))
        self.assertTrue(st["swing"]["checks"][0]["ok"])  # 上升趨勢：中期趨勢符合
        self.assertEqual(strategy.status(px(-1.0))["trend"]["label"], "偏弱")


class Page(unittest.TestCase):
    def test_trend_tab_order(self):
        body = HTML[HTML.index('<div class="tabp" data-t="trend">'):HTML.index('<div class="tabp" data-t="fin">')]
        order = [body.index(x) for x in ("trendCard(s, d)", "timelineCard(s, d)", "decisionCard(s, d)")]
        self.assertEqual(order, sorted(order))
        f = HTML[HTML.index("function trendCard"):HTML.index("function drawTrend")]
        order = [f.index(x) for x in ('id="hzSeg"', 'id="tc"', 'id="t-ck"', 'id="t-plan"')]
        self.assertEqual(order, sorted(order))
        self.assertIn("目前尚未觸發", f)
        self.assertIn("研究試算價位", f)  # 試算價位收合、標示非當前訊號
        self.assertIn("不是 AI 預測，也不是買進建議", f)
        self.assertIn('id="t-adv"', HTML)  # 進階研究收合

    def test_horizon_states_are_specific(self):
        for lab in ("符合進場條件", "趨勢偏多，尚未觸發進場", "風險偏高", "趨勢偏弱", "資料不足"):
            self.assertIn(lab, HTML)
        self.assertNotIn("上漲機率", HTML[HTML.index("function trendCard"):HTML.index("function drawTrend")])
