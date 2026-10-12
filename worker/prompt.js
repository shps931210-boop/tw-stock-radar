// 把網站上的個股資料整理成給 AI 看的資料包，以及檢查使用者送來的對話。
// 沒有用到任何外部套件，可以直接用 node --test 測試。

export const MAX_TURNS = 6;      // 一次對話最多保留幾輪
export const MAX_CHARS = 500;    // 每個問題最多幾個字

export const SYSTEM = `你是「股栗子」網站的 AI 個股助理，用繁體中文回答台股問題。

規則：
- 只根據下方提供的這檔股票資料回答。資料裡沒有的，直接說「網站目前沒有這項資料」，不要猜，也不要用你記得的舊資料。
- 引用數字時，說明是哪個日期或哪一季的資料。
- 用一般人聽得懂的白話，先講結論，再講理由；專有名詞第一次出現時用一句話解釋。
- 可以說明這檔股票的性質（偏長期體質、成長、價值、景氣循環、短線動能等）、財報與營收的重點、風險在哪裡，但不要叫使用者買進或賣出，也不要給目標價或預測股價。
- 網站的分數是全市場百分位（0～100，50 是中位數），不是上漲機率。
- 回答控制在 300 字以內，除非使用者要求詳細說明。
- 最後一行固定寫：「以上根據股栗子網站的公開資料整理，不是投資建議。」`;

const ROW_KEYS = [
  "stock_id", "name", "industry", "market", "date", "close", "chg_pct", "market_cap",
  "per", "pbr", "dy", "per_pctile_1y", "eps_ttm", "eps_ttm_yoy", "eps_ttm_turn", "roe", "roa", "roe_years_15",
  "gross_margin", "op_margin", "debt_ratio", "ocf_to_ni", "fcf_yield", "fscore", "div_years", "payout",
  "rev_ym", "rev_yoy", "rev_yoy_3m", "rev_ytd_yoy", "fin_period",
  "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_1y", "mdd_1y", "beta", "ma_bull", "rsi",
  "inst_date", "foreign_5d", "trust_5d", "foreign_streak", "trust_streak", "margin_chg_20d",
  "score_quality", "score_growth", "score_value", "score_momentum", "score_trend",
  "industry_rank", "industry_n", "complete", "missing_detail", "stale_notes", "exclude_reasons",
];

const pick = (o, keys) => Object.fromEntries(keys.filter((k) => o?.[k] != null).map((k) => [k, o[k]]));

/** 從網站的 data/stock/代號.json 整理出精簡的資料包（約幾千字），避免把整年股價都送給 AI。 */
export function buildContext(d) {
  const row = d.row || {};
  const prices = (d.prices || []).slice(-60).map((p) => [p.date, p.close, Math.round((p.value || 0) / 1e6)]);
  const ctx = {
    "基本資料與指標": pick(row, ROW_KEYS),
    "AI 解讀": row.ai ? { "標籤": row.ai.tag, "優勢": row.ai.strengths, "風險": row.ai.risks, "摘要": row.ai.summary,
      "各模式分數與名次": row.ai.modes } : undefined,
    "技術訊號": row.timing ? { "狀態": row.timing.status, "訊號": row.timing.triggers, "過熱提醒": row.timing.hot } : undefined,
    "個股性質卡": d.profile,
    "近 13 個月營收（元）": (d.revenue || []).slice(-13),
    "近 8 季財報（元）": (d.financials || []).slice(-8).map((f) => pick(f, ["period", "revenue", "gross_profit", "operating_income", "net_income", "eps", "ocf", "debt_ratio", "total_assets", "total_liabilities", "equity"])),
    "股利（每股，元）": (d.dividends || []).slice(-6),
    "近 60 個交易日收盤價 [日期, 收盤, 成交金額百萬]": prices,
    "近 20 日三大法人買賣超（股）": (d.institutional || []).slice(-20),
  };
  return JSON.stringify(ctx, (k, v) => (typeof v === "number" ? Math.round(v * 100) / 100 : v));
}

/** 檢查前端送來的對話：只留 user／assistant、字數限制、最後一則一定是使用者的問題。 */
export function cleanMessages(messages) {
  if (!Array.isArray(messages) || !messages.length) throw new Error("沒有問題");
  const out = messages
    .filter((m) => (m?.role === "user" || m?.role === "assistant") && typeof m.content === "string" && m.content.trim())
    .slice(-MAX_TURNS * 2)
    .map((m) => ({ role: m.role, content: m.content.slice(0, m.role === "user" ? MAX_CHARS : 4000) }));
  while (out.length && out[0].role !== "user") out.shift();
  if (!out.length || out[out.length - 1].role !== "user") throw new Error("最後一則要是問題");
  return out;
}

export const validId = (sid) => typeof sid === "string" && /^[0-9]{4}[A-Z]?$/.test(sid);
