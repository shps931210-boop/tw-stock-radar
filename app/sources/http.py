"""共用的 HTTP 請求：暫時性錯誤自動重試，錯誤訊息保留 HTTP 狀態碼與失敗類型。"""
from __future__ import annotations

import time

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


def get(session: requests.Session, url: str, *, tries: int = 3, backoff: float = 2.0, **kw) -> requests.Response:
    """連線失敗、逾時、429、5xx 會等 2、4 秒後重試；其他 4xx（例如 403 被擋、404 端點變更）立刻失敗。"""
    kw.setdefault("timeout", 30)
    for i in range(tries):
        try:
            r = session.get(url, **kw)
            if r.status_code in RETRY_STATUS and i < tries - 1:
                time.sleep(backoff * 2 ** i)
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if i == tries - 1:
                raise
            time.sleep(backoff * 2 ** i)
    raise RuntimeError("unreachable")


def describe(e: Exception) -> str:
    """把例外轉成看得懂的一行：HTTP 狀態碼、DNS 失敗、被代理擋、逾時、格式變更。"""
    if isinstance(e, requests.HTTPError) and e.response is not None:
        r = e.response
        return f"HTTP {r.status_code} {r.reason or ''}（{r.url.split('?')[0]}）".strip()
    msg = str(e)
    if isinstance(e, requests.Timeout):
        return "連線逾時"
    if isinstance(e, requests.ConnectionError):
        if "NameResolution" in msg or "Name or service not known" in msg or "getaddrinfo" in msg:
            return "連線失敗：找不到網域（DNS 無法解析，通常是執行環境沒有對外網路）"
        if "Tunnel connection failed" in msg or "ProxyError" in msg:
            code = next((c for c in ("403", "407", "502") if c in msg), "")
            return f"連線失敗：被網路代理拒絕{(' ' + code) if code else ''}（執行環境的網路政策限制）"
        return "連線失敗：" + msg[:120]
    if isinstance(e, (ValueError, KeyError)):
        return "回應格式不符（API 可能改版）：" + msg[:160]
    return f"{type(e).__name__}: {msg[:160]}"
