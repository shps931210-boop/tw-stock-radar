"""SQLite 股票資料庫。所有資料來源都寫進這裡，分析與網站只讀這裡。"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from .config import DATA_DIR
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (stock_id TEXT PRIMARY KEY, name TEXT, market TEXT, industry TEXT);
CREATE TABLE IF NOT EXISTS prices (stock_id TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
  volume REAL, value REAL, PRIMARY KEY (stock_id, date));
CREATE TABLE IF NOT EXISTS index_prices (date TEXT PRIMARY KEY, close REAL);
CREATE TABLE IF NOT EXISTS institutional (stock_id TEXT, date TEXT, foreign_net REAL, trust_net REAL,
  dealer_net REAL, PRIMARY KEY (stock_id, date));
CREATE TABLE IF NOT EXISTS margin (stock_id TEXT, date TEXT, margin_bal REAL, short_bal REAL,
  PRIMARY KEY (stock_id, date));
CREATE TABLE IF NOT EXISTS valuation (stock_id TEXT, date TEXT, per REAL, pbr REAL, dy REAL,
  PRIMARY KEY (stock_id, date));
CREATE TABLE IF NOT EXISTS revenue (stock_id TEXT, ym TEXT, revenue REAL, PRIMARY KEY (stock_id, ym));
CREATE TABLE IF NOT EXISTS financials (stock_id TEXT, period TEXT, period_end TEXT, revenue REAL,
  gross_profit REAL, operating_income REAL, net_income REAL, eps REAL, total_assets REAL,
  total_liabilities REAL, equity REAL, current_assets REAL, current_liabilities REAL, share_capital REAL,
  ocf REAL, capex REAL, PRIMARY KEY (stock_id, period));
CREATE TABLE IF NOT EXISTS dividends (stock_id TEXT, year INTEGER, cash REAL, stock REAL,
  PRIMARY KEY (stock_id, year));
CREATE TABLE IF NOT EXISTS news (stock_id TEXT, date TEXT, title TEXT, source TEXT, link TEXT,
  PRIMARY KEY (stock_id, title));
CREATE TABLE IF NOT EXISTS disposition (stock_id TEXT PRIMARY KEY, period TEXT, reason TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS fetch_log (source TEXT, key TEXT, fetched_at TEXT, ok INTEGER,
  PRIMARY KEY (source, key));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def path(name: str = "stocks.db"):
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / name


@contextmanager
def connect(name: str = "stocks.db"):
    con = sqlite3.connect(path(name), timeout=30)
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert(con, table: str, rows: list[dict]):
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    con.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    return len(rows)


def read(con, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def log_fetch(con, source: str, key: str, ok: bool = True):
    con.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?)",
                (source, key, config.now().isoformat(timespec="seconds"), int(ok)))


def fetched_at(con, source: str, key: str) -> datetime | None:
    row = con.execute("SELECT fetched_at FROM fetch_log WHERE source=? AND key=?", (source, key)).fetchone()
    return datetime.fromisoformat(row[0]) if row else None


def set_meta(con, k: str, v: str):
    con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))


def get_meta(con, k: str, default=None):
    row = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return row[0] if row else default
