"""測試分組（V20.1）：

  python -m tests.run unit          純單元測試：不准連外網（連線會直接失敗），每個測試最多 60 秒，全部約 1 分鐘
  python -m tests.run integration   整合測試：讀真實官方 OpenAPI、正式資料庫（data/stocks.db），需要網路與資料庫，時間較長
  python -m tests.run all           兩組都跑

單元測試會連網就是測試寫錯了（應該用假資料），所以直接擋掉並指出是哪個測試，不會卡在逾時重試。
網站畫面驗收（瀏覽器）另外在 tests/e2e/home.mjs，部署後由 e2e.yml 執行。
"""
from __future__ import annotations

import signal
import socket
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PER_TEST = 60   # 單一單元測試的時間上限（秒）


class NetworkBlocked(RuntimeError):
    pass


def block_network() -> None:
    real = socket.socket.connect

    def guard(self, addr, *a, **k):
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if host in ("127.0.0.1", "localhost", "::1") or self.family == getattr(socket, "AF_UNIX", None):
            return real(self, addr, *a, **k)
        raise NetworkBlocked(f"單元測試不允許連外網：{host}（請改用假資料，或把測試移到 tests/integration）")

    socket.socket.connect = guard
    socket.create_connection = lambda addr, *a, **k: (_ for _ in ()).throw(NetworkBlocked(f"單元測試不允許連外網：{addr[0]}"))


class Timed(unittest.TextTestResult):
    slow: list = []

    def startTest(self, test):
        self._t = time.time()
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError(f"超過 {PER_TEST} 秒：{test.id()}")))
            signal.alarm(PER_TEST)
        super().startTest(test)

    def stopTest(self, test):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        Timed.slow.append((time.time() - self._t, test.id()))
        super().stopTest(test)


def suite(group: str) -> unittest.TestSuite:
    """unit：tests/ 底下全部（tests/integration 會整組略過，因為沒有 INTEGRATION=1）；integration：只跑 tests/integration；all：全部。"""
    import os
    loader = unittest.TestLoader()
    if group == "unit":
        os.environ.pop("INTEGRATION", None)
        return loader.discover(str(HERE), pattern="test_*.py", top_level_dir=str(HERE.parent))
    os.environ["INTEGRATION"] = "1"
    return loader.discover(str(HERE / "integration") if group == "integration" else str(HERE), pattern="test_*.py", top_level_dir=str(HERE.parent))


def main(argv: list[str]) -> int:
    group = argv[0] if argv else "unit"
    if group not in ("unit", "integration", "all"):
        print(__doc__)
        return 2
    if group == "unit":
        block_network()
    t0 = time.time()
    r = unittest.TextTestRunner(resultclass=Timed, verbosity=1).run(suite(group))
    print(f"\n{group}：{r.testsRun} 個測試、失敗 {len(r.failures)}、錯誤 {len(r.errors)}、略過 {len(r.skipped)}，共 {time.time() - t0:.1f} 秒")
    print("最慢的 5 個：" + "；".join(f"{i.split('.')[-1]} {t:.1f}s" for t, i in sorted(Timed.slow, reverse=True)[:5]))
    return 0 if r.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
