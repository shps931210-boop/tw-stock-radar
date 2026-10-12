"""智慧警報：同一組案例（tests/alert_cases.json）測後端 app/alerts.py 和網頁的 alertEvents（用 Node 執行），兩邊結果必須完全一樣。"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from app import alerts

ROOT = Path(__file__).parent.parent
CASES = json.loads((ROOT / "tests" / "alert_cases.json").read_text("utf-8"))
HTML = (ROOT / "static" / "index.html").read_text("utf-8")
CORE = re.search(r"const ALERT_TYPES = \{.*?^\};", HTML, re.S | re.M).group(0) + "\n" + \
    re.search(r"// <alert-core>(.*?)// </alert-core>", HTML, re.S).group(1)


def run_js(c):
    args = ", ".join(json.dumps(c[k], ensure_ascii=False) for k in ("snap", "watch", "rules", "state", "today"))
    out = subprocess.run(["node", "-e", CORE + f"\nconsole.log(JSON.stringify(alertEvents({args})));"], capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    return r["notes"], r["state"]


class Backend(unittest.TestCase):
    def test_cases(self):
        for c in CASES:
            with self.subTest(c["name"]):
                notes, state = alerts.events(c["snap"], c["watch"], c["rules"], c["state"], c["today"])
                self.assertEqual(sorted(n["key"] for n in notes), c["keys"])
                if "state_out" in c:
                    self.assertEqual(state, c["state_out"])

    def test_run_writes_only_through_engine(self):
        c = CASES[1]

        class Fake:
            def __init__(self):
                self.writes = []
            def get(self, table, select):
                return {"user_state": [{"user_id": "u1", "watch": {k: {} for k in c["watch"]}}, {"user_id": "u2", "watch": {}}],
                        "alert_rules": [], "alert_processing_state": [{"user_id": "u1", "data_date": c["state"]["date"], "baseline": c["state"]["base"]}]}[table]
            def insert(self, table, rows, on_conflict, merge):
                self.writes.append((table, on_conflict, merge, rows))

        f = Fake()
        res = alerts.run(c["snap"], f, c["today"])
        self.assertEqual(res["users"], 1)
        (t1, oc1, m1, notes), (t2, oc2, m2, states) = f.writes
        self.assertEqual((t1, oc1, m1), ("user_notifications", "user_id,event_key", False))   # 重複的通知直接略過
        self.assertEqual(sorted(n["event_key"] for n in notes), c["keys"])
        self.assertEqual((t2, oc2, m2, states[0]["data_date"]), ("alert_processing_state", "user_id", True, "2026-10-12"))

    def test_no_secret_skips(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if k != "SUPABASE_SECRET_KEY"}
            old = os.environ.copy()
            os.environ.clear(); os.environ.update(env)
            try:
                alerts.main(["x", f"{d}/engine.json"])
            finally:
                os.environ.clear(); os.environ.update(old)
            self.assertFalse(json.loads(Path(d, "engine.json").read_text("utf-8"))["enabled"])


@unittest.skipUnless(shutil.which("node"), "需要 Node")
class SameAsWebsite(unittest.TestCase):
    def test_js_matches_backend(self):
        for c in CASES:
            with self.subTest(c["name"]):
                js_notes, js_state = run_js(c)
                py_notes, py_state = alerts.events(c["snap"], c["watch"], c["rules"], c["state"], c["today"])
                self.assertEqual(js_notes, py_notes)
                self.assertEqual(js_state, py_state)


if __name__ == "__main__":
    unittest.main()


class Rest(unittest.TestCase):
    def test_request_shape(self):
        class R:
            def __init__(self, data=None): self.data = data or []
            def raise_for_status(self): pass
            def json(self): return self.data
        class S:
            def __init__(self): self.headers, self.calls = {}, []
            def get(self, url, **kw): self.calls.append(("GET", url, kw)); return R([{"user_id": "a"}])
            def post(self, url, **kw): self.calls.append(("POST", url, kw)); return R()
        s = S()
        sp = alerts.Supa("https://x.supabase.co/", "sb_secret_abc", s)
        self.assertEqual(sp.get("user_state", "user_id,watch"), [{"user_id": "a"}])
        sp.insert("user_notifications", [{"a": 1}], "user_id,event_key", merge=False)
        self.assertEqual(s.headers["apikey"], "sb_secret_abc")
        self.assertNotIn("Authorization", s.headers)          # 新版 secret key 不是 JWT
        g, p = s.calls
        self.assertEqual(g[1], "https://x.supabase.co/rest/v1/user_state")
        self.assertEqual(p[2]["params"], {"on_conflict": "user_id,event_key"})
        self.assertIn("resolution=ignore-duplicates", p[2]["headers"]["Prefer"])
