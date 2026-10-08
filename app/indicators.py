"""常用技術指標（純 pandas 實作，不依賴 TA-Lib）。"""
import pandas as pd


def ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, 1e-9))


def kd(df: pd.DataFrame, n: int = 9) -> tuple[pd.Series, pd.Series]:
    """台灣常用的 KD(9,3,3)：K = 2/3 K昨 + 1/3 RSV。"""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = ((df["close"] - low_n) / (high_n - low_n).replace(0, 1e-9) * 100).fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return k, d


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    return dif, dea, dif - dea


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = ma(close, n)
    std = close.rolling(n).std()
    return mid + k * std, mid, mid - k * std


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """在價格表上加上所有指標欄位。"""
    df = df.copy()
    c = df["close"]
    for n in (5, 10, 20, 60, 120):
        df[f"ma{n}"] = ma(c, n)
    df["vma20"] = ma(df["volume"], 20)
    df["rsi"] = rsi(c)
    df["k"], df["d"] = kd(df)
    df["dif"], df["dea"], df["hist"] = macd(c)
    df["bb_up"], _, df["bb_low"] = bollinger(c)
    return df
