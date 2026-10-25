"""V12 介面：主導覽五項、首頁五個區塊、個股頁先給判斷、沒有驗證過的價位不顯示。"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).parent.parent / "static" / "index.html").read_text("utf-8")


def fn(name):
    return re.search(rf"function {name}\(.*?\n}}\n", HTML, re.S).group(0)


class Nav(unittest.TestCase):
    def test_five_main_items(self):
        nav = re.search(r'<nav id="nav".*?</nav>', HTML, re.S).group(0)
        top = re.findall(r'<a href="(#/[^"]*)" data-r="[^"]*" class="navmain"', nav)
        self.assertEqual(top, ["#/", "#/ai", "#/search", "#/watch", "#/performance"])
        bnav = re.search(r'<div class="bnav".*?</div>', HTML, re.S).group(0)
        self.assertEqual(len(re.findall(r"<a |<button ", bnav)), 5)
        self.assertIn("repeat(5, 1fr)", HTML)

    def test_every_page_still_reachable(self):
        nav = re.search(r'<nav id="nav".*?</nav>', HTML, re.S).group(0)
        for r in ["screener", "ask", "market", "sectors", "news", "buy", "calendar", "research", "me", "mine", "portfolio", "account", "alerts", "trust", "catalog", "privacy"]:
            self.assertIn(f'href="#/{r}"', nav, r)
        for r in ["research: pageResearch", "search: pageSearch"]:
            self.assertIn(r, HTML)


class Home(unittest.TestCase):
    def test_four_blocks_search_first(self):
        home = fn("pageHome")
        order = [home.index(x) for x in ['id="hs"', 'id="hmkt"', "homePicks()", "homeWatch()"]]
        self.assertEqual(order, sorted(order))                   # V16：搜尋 → 今日大盤 → AI 精選（分數）→ 自選股
        for x in ["briefCard()", "statusCard()", "changesCard()", "loadNews", "homePerf()", "drawTreemap"]:
            self.assertNotIn(x, home)

    def test_moved_cards_still_shown_elsewhere(self):
        self.assertIn("${briefCard()}", fn("pageMarket"))
        self.assertIn("${statusCard()}", fn("pageTrust"))
        self.assertIn("${changesCard()}", fn("pageAI"))
        self.assertIn("perfTable()", fn("pagePerformance"))

    def test_pick_row_has_one_reason(self):
        self.assertNotIn("留意", fn("pickRow"))


class Stock(unittest.TestCase):
    def test_quick_judgment_comes_first(self):
        st = fn("pageStock")
        self.assertLess(st.index("${quickCard(s, d)}"), st.index('class="tabp"'))

    def test_price_zone_only_when_validated(self):
        q = fn("quickCard")
        self.assertIn("尚無有效參考區間", q)
        self.assertIn("bt.excess > 0", q)          # 規則回測要勝過 0050 才顯示價位
        self.assertNotIn("上漲機率", q)
        for x in ["參考價位", "主要風險", "建議觀察期間", "過去實績", "加入自選股", 'data-go="trend"', 'data-go="ai"', "設定提醒"]:
            self.assertIn(x, q)

    def test_four_tabs(self):
        st = fn("pageStock")
        self.assertEqual(re.findall(r'class="tabp" data-t="(\w+)"', st), ["ov", "trend", "fin", "ai"])
        self.assertIn("${aiHistCard(s, d)}", st)


if __name__ == "__main__":
    unittest.main()


class HomeV16(unittest.TestCase):
    """V16：大盤資訊與 AI 分數要清楚可見（使用者說上一版比較清楚）。"""
    def test_market_card_has_index_0050_breadth_strength(self):
        home = fn("pageHome")
        for x in ["加權指數", "0050", "市場強弱", "上漲", "下跌", "漲停", "mk-bar"]:
            self.assertIn(x, home)

    def test_pick_row_shows_numeric_score_status_risk(self):
        row = fn("pickRow")
        for x in ["a.score", "/100", "AI 評分", "rating(s)", "riskLevel(s)", "查看分析"]:
            self.assertIn(x, row)
        self.assertNotIn("機率", row)   # 分數不是上漲機率

    def test_pick_row_can_add_to_watchlist(self):
        self.assertIn("starBtn(s.stock_id)", fn("pickRow"))
        self.assertIn("e.preventDefault(); toggleWatch", HTML)    # 按星號不會跳進個股頁

    def test_watch_row_shows_price_score_risk_rank_move(self):
        w = fn("homeWatch")
        for x in ["s.close", "s.chg_pct", "a.score", "AI 評分", "riskLevel(s)", "rankMove(s)", "留意："]:
            self.assertIn(x, w)

    def test_market_card_says_data_time_not_realtime(self):
        home = fn("pageHome")
        self.assertIn("網站更新", home)
        self.assertIn("非即時行情", home)

    def test_home_regression_e2e_exists(self):
        e2e = (Path(__file__).parent / "e2e" / "home.mjs").read_text("utf-8")
        for x in ["大盤資訊清楚可見", "AI 數字評分清楚可見", "股票搜尋正常", "AI 三種策略可以切換", "自選股可以加入並顯示",
                  "漲跌停資訊正常", "個股分析正常", "AI 績效資料正常", "版面沒有橫向超出", "資料時間與盤後狀態正確"]:
            self.assertIn(x, e2e)
        wf = (Path(__file__).parent.parent / ".github" / "workflows" / "e2e.yml").read_text("utf-8")
        self.assertIn("tests/e2e/home.mjs", wf)
