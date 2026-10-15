"""我的投資組合：static/portfolio.js 的計算測試（需要 node；沒有 node 時略過）。"""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class PortfolioJS(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "沒有安裝 node")
    def test_calculations(self):
        r = subprocess.run(["node", str(ROOT / "tests" / "portfolio.test.cjs")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

    def test_site_ships_module(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        self.assertIn('<script src="portfolio.js"></script>', html)
        self.assertIn("portfolio.js", (ROOT / "build_site.py").read_text("utf-8"))
        self.assertIn("add column if not exists portfolio", (ROOT / "supabase" / "schema.sql").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
