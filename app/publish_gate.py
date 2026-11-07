"""到期績效發布閘門（2026-11-07，交接企劃書 GZ-P0-03）。

之前 build_site.py 只在「reconcile.json 存在而且 all_ok 不是真」時才擋：檔案不存在、JSON 壞掉時反而會放行。
現在改成 Fail-Closed：下面每一項都要通過才發布到期績效（maturity/*.json、fair.json），任何一項不通過就只發布 blocked 索引。

1. reconcile.json 存在，而且是合法 JSON 物件
2. all_ok 必須是布林值 True（不接受 "true"、1 之類的真值）；violations 必須是整數 0
3. 對帳時間不能比到期索引舊（對帳步驟當掉時留下的舊結果不能當作通過）
4. 對帳涵蓋的批次檔和目前所有到期批次檔完全相同（不能少、也不能多）
5. 每個批次檔都標 ok=True，而且檔案內容的 SHA-256 和對帳當時一致（對帳後被改過就不發布）
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BATCH_GLOB = "????-??-??_h*.json"


def check(rc_path: Path, mat_dir: Path) -> dict:
    """回傳 {"ok": bool, "why": [原因...], "files": [有問題的檔案...], "checked_at": 對帳時間}。"""
    why, bad = [], []
    if not rc_path.exists():
        return {"ok": False, "why": ["找不到獨立對帳結果 reconcile.json"], "files": [], "checked_at": None}
    try:
        rc = json.loads(rc_path.read_text("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return {"ok": False, "why": [f"reconcile.json 格式錯誤：{str(e)[:80]}"], "files": [], "checked_at": None}
    if not isinstance(rc, dict):
        return {"ok": False, "why": ["reconcile.json 不是物件"], "files": [], "checked_at": None}
    at = rc.get("generated_at")
    if rc.get("all_ok") is not True:
        why.append(f"all_ok 不是 True（{rc.get('all_ok')!r}）")
    v = rc.get("violations")
    if not isinstance(v, int) or isinstance(v, bool) or v != 0:
        why.append(f"不合格批次違規數不是 0（{v!r}）")
    idx = mat_dir / "index.json"
    if idx.exists():
        try:
            ia = json.loads(idx.read_text("utf-8")).get("generated_at") or ""
        except ValueError:
            ia = ""
            why.append("到期索引 index.json 格式錯誤")
        if not isinstance(at, str) or at < ia:
            why.append(f"對帳結果（{at}）比到期索引（{ia}）舊，這次對帳沒有完成")
    files = rc.get("files")
    if not isinstance(files, list):
        why.append("reconcile.json 沒有批次清單")
        files = []
    have = {p.name: p for p in mat_dir.glob(BATCH_GLOB)}
    rec = {f.get("file"): f for f in files if isinstance(f, dict)}
    missing, extra = sorted(set(have) - set(rec)), sorted(set(rec) - set(have))
    if missing:
        why.append(f"{len(missing)} 個到期批次沒有對帳")
        bad += missing
    if extra:
        why.append(f"對帳清單有 {len(extra)} 個不存在的批次")
        bad += extra
    for name in sorted(set(have) & set(rec)):
        f = rec[name]
        if f.get("ok") is not True:
            bad.append(name)
            why.append(f"{name} 對帳不通過")
            continue
        sha = f.get("sha256")
        if not sha or hashlib.sha256(have[name].read_bytes()).hexdigest() != sha:
            bad.append(name)
            why.append(f"{name} 對帳後內容有變動（或對帳沒有記錄指紋）")
    return {"ok": not why, "why": why, "files": bad, "checked_at": at if isinstance(at, str) else None}
