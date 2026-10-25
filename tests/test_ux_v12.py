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
    def test_five_blocks(self):
        home = fn("pageHome")
        for x in ['id="hmkt"', "homePicks()", "homeWatch()", 'id="hheat"', "homePerf()"]:
            self.assertIn(x, home)
        for x in ["briefCard()", "statusCard()", "changesCard()", "loadNews"]:   # 搬到各自的頁面
            self.assertNotIn(x, home)

    def test_moved_cards_still_shown_elsewhere(self):
        self.assertIn("${briefCard()}", fn("pageMarket"))
        self.assertIn("${statusCard()}", fn("pageTrust"))
        self.assertIn("${changesCard()}", fn("pageAI"))


class Stock(unittest.TestCase):
    def test_quick_judgment_comes_first(self):
        st = fn("pageStock")
        self.assertLess(st.index("${quickCard(s, d)}"), st.index('class="tabp"'))

    def test_price_zone_only_when_validated(self):
        q = fn("quickCard")
        self.assertIn("尚無有效參考區間", q)
        self.assertIn("bt.excess > 0", q)          # 規則回測要勝過 0050 才顯示價位
        self.assertNotIn("上漲機率", q)
        for x in ["參考價位", "主要風險", "建議觀察期間", "過去實績"]:
            self.assertIn(x, q)


if __name__ == "__main__":
    unittest.main()
