"""V11：模型指紋只看計算邏輯、文案一致性。"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from app import track

ROOT = Path(__file__).parent.parent


class ModelHash(unittest.TestCase):
    def test_wording_does_not_change_hash_but_logic_does(self):
        base = 'T={"quality": "獲利穩定"}\ndef f(r):\n    """說明"""\n    return r["roe"] > 15  # 註解'
        self.assertEqual(track._logic(base), track._logic(base.replace("獲利穩定", "獲利能力穩定").replace("說明", "新說明").replace("  # 註解", "")))
        self.assertNotEqual(track._logic(base), track._logic(base.replace("> 15", "> 16")))
        self.assertNotEqual(track._logic(base), track._logic(base.replace('"roe"', '"roa"')))


class Wording(unittest.TestCase):
    """術語表：舊名稱不應再出現在網站上（README 的術語表除外）。"""
    OLD = ["觀察清單", "我的排行", "今日技術訊號", "成績單", "可信度中心", "策略雷達", "AI 選股", "我的儀表板", "尚未經台股歷史回測驗證",
           "多因子篩選器", "個人化選股", "投資儀表板", "數據品質與來源", "績效驗證"]  # V15 改名

    def test_no_old_terms_in_site(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        for t in self.OLD:
            self.assertNotIn(t, html, t)

    def test_nav_groups_have_descriptions(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        nav = re.search(r'<nav id="nav".*?</nav>', html, re.S).group(0)
        for a in re.findall(r'<div class="menu">(.*?)</div>', nav, re.S):
            for link in re.findall(r"<a [^>]*>.*?</a>", a, re.S):
                self.assertIn("<small>", link)


if __name__ == "__main__":
    unittest.main()
