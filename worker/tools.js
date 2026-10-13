// AI 研究助理的資料工具：AI 依問題決定要查什麼，這裡從股栗子網站的公開 JSON 取資料。
// 每個工具只回傳精簡結果（不會把整份資料丟給 AI），並附上資料日期與來源，讓 AI 回答時能引用。
// 不依賴外部套件，可以用 node --test 測試（load 由呼叫端提供）。

const ID = /^[0-9]{4}[A-Z]?$/;
const NUM_OPS = { ">=": (a, b) => a >= b, "<=": (a, b) => a <= b, ">": (a, b) => a > b, "<": (a, b) => a < b };
const r2 = (v) => (typeof v === "number" ? Math.round(v * 100) / 100 : v);

export const TOOLS = [
  {
    name: "find_stocks",
    description: "用公司名稱或代號找股票代號。使用者說公司名稱（例如「台積電」「聯發科」）時先用這個確認代號。",
    input_schema: { type: "object", properties: { query: { type: "string", description: "名稱或代號，可以只寫部分名稱" } }, required: ["query"] },
  },
  {
    name: "get_stock_detail",
    description: "查一檔股票的完整研究資料：基本指標、量化評分拆解、近 13 個月營收、近 8 季財報、股利、近 60 日股價、法人買賣超、最新公告與新聞。分析單一公司時使用。",
    input_schema: { type: "object", properties: { stock_id: { type: "string", description: "4 位數股票代號，例如 2330" } }, required: ["stock_id"] },
  },
  {
    name: "compare_stocks",
    description: "並列比較 2～6 檔股票的估值、獲利、成長、股價表現、法人與量化評分。",
    input_schema: { type: "object", properties: { stock_ids: { type: "array", items: { type: "string" }, minItems: 2, maxItems: 6 } }, required: ["stock_ids"] },
  },
  {
    name: "screen_stocks",
    description: "依條件篩選全市場普通股。可用的 metric 見 explain_metric 或欄位清單：per、pbr、dy、roe、roa、gross_margin、op_margin、debt_ratio、rev_yoy、rev_yoy_3m、rev_ytd_yoy、eps_ttm、eps_ttm_yoy、ret_5d、ret_20d、ret_3m、ret_12m、vol_1y、mdd_1y、market_cap、avg_value_20、score、rank、score_quality、score_growth、score_value、score_momentum、score_trend、foreign_streak、trust_streak、fscore、div_years。百分比欄位的單位是 %（例如 rev_yoy 20 代表年增 20%）。",
    input_schema: {
      type: "object",
      properties: {
        filters: { type: "array", items: { type: "object", properties: { metric: { type: "string" }, op: { type: "string", enum: [">=", "<=", ">", "<"] }, value: { type: "number" } }, required: ["metric", "op", "value"] } },
        industry: { type: "string", description: "只看某個產業（證交所分類名稱，例如「半導體業」）" },
        sort_by: { type: "string", description: "排序欄位，預設 score" },
        order: { type: "string", enum: ["desc", "asc"] },
        limit: { type: "integer", minimum: 1, maximum: 20 },
      },
    },
  },
  {
    name: "market_overview",
    description: "最新收盤的大盤概況：加權指數、漲跌家數、各產業近期報酬、AI 量化評分前 20 名的變化、未來兩週的除權息與財報事件。",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "industry_report",
    description: "一個產業的概況：產業報酬、成分股數、依量化評分排序的主要公司、最近的公告與新聞。",
    input_schema: { type: "object", properties: { industry: { type: "string", description: "產業名稱，例如「航運業」「半導體業」" } }, required: ["industry"] },
  },
  {
    name: "search_news",
    description: "搜尋最近 7 天的上市櫃公司重大訊息與財經新聞（標題、摘要、來源、時間、連結）。可以依股票、產業、關鍵字、類型篩選。",
    input_schema: {
      type: "object",
      properties: {
        stock_ids: { type: "array", items: { type: "string" } },
        industry: { type: "string" },
        keyword: { type: "string" },
        kind: { type: "string", enum: ["announcement", "news", "all"] },
        limit: { type: "integer", minimum: 1, maximum: 20 },
      },
    },
  },
  {
    name: "performance_summary",
    description: "股栗子選股名單的事後績效：AI 前 20 名、技術訊號股與 0050 比較的超額報酬、勝率、紀錄天數與限制。",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "explain_metric",
    description: "查網站指標的定義、公式、單位與限制（例如本益比、ROE、F-Score、量化評分）。",
    input_schema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
].map((t) => ({ ...t, eager_input_streaming: true }));

const TOOL_LABEL = {
  find_stocks: (i) => `尋找股票「${i.query}」`, get_stock_detail: (i) => `讀取 ${i.stock_id} 的研究資料`,
  compare_stocks: (i) => `比較 ${(i.stock_ids || []).join("、")}`, screen_stocks: () => "依條件篩選股票",
  market_overview: () => "讀取大盤與產業概況", industry_report: (i) => `讀取「${i.industry}」產業資料`,
  search_news: (i) => `搜尋公告與新聞${i.keyword ? `「${i.keyword}」` : ""}`, performance_summary: () => "讀取選股績效紀錄",
  explain_metric: (i) => `查指標說明「${i.query}」`,
};
export const toolLabel = (name, input) => (TOOL_LABEL[name] ? TOOL_LABEL[name](input || {}) : name);

// ---- 全市場資料（data/ai/universe.json，欄位 → 一整列） ----
function table(u) {
  const ids = u.cols.stock_id, at = new Map(ids.map((id, i) => [id, i]));
  const row = (i, keys = u.fields) => Object.fromEntries(keys.map((k) => [k, r2((u.cols[k] || [])[i])]).filter(([, v]) => v != null && v !== ""));
  return { ids, at, row, col: (k) => u.cols[k] || [] };
}
const BRIEF = ["stock_id", "name", "industry", "close", "chg_pct", "ret_20d", "per", "dy", "roe", "rev_yoy", "score", "rank", "tag"];
const COMPARE = ["stock_id", "name", "industry", "date", "close", "chg_pct", "ret_20d", "ret_3m", "ret_12m", "market_cap", "per", "pbr", "dy", "per_pctile_1y",
  "eps_ttm", "eps_ttm_yoy", "roe", "gross_margin", "op_margin", "debt_ratio", "fscore", "rev_ym", "rev_yoy", "rev_yoy_3m", "fin_period", "vol_1y", "mdd_1y",
  "foreign_5d", "trust_5d", "score", "rank", "score_quality", "score_growth", "score_value", "score_momentum", "score_trend", "industry_rank", "industry_n", "tag", "signal"];

function cleanIds(xs, max) {
  return [...new Set((Array.isArray(xs) ? xs : []).map((x) => String(x).trim().toUpperCase()).filter((x) => ID.test(x)))].slice(0, max);
}

/** load(path) 回傳網站上 data/<path> 的 JSON（呼叫端負責快取）。回傳 run(name, input) → { result, sources } */
export function makeRunner(load) {
  let U = null;
  const uni = async () => {
    if (!U) { const u = await load("ai/universe.json"); U = { u, t: table(u) }; }
    return U;
  };
  const dates = (u) => ({ 資料日期: u.data_dates, 模型版本: u.model, 說明: u.note });
  const SRC_U = { name: "股栗子全市場指標", path: "data/ai/universe.json" };

  const run = {
    async find_stocks({ query }) {
      const { u, t } = await uni(), q = String(query || "").trim();
      if (!q) return { error: "請提供名稱或代號" };
      const names = t.col("name");
      const hits = t.ids.map((id, i) => [id, i]).filter(([id, i]) => id === q.toUpperCase() || id.startsWith(q) || (names[i] || "").includes(q)).slice(0, 10);
      return { result: { 符合: hits.map(([, i]) => t.row(i, ["stock_id", "name", "industry", "market"])), ...(hits.length ? {} : { 說明: "網站資料裡找不到，可能是 ETF、下市或名稱不同" }) }, sources: [SRC_U], u };
    },
    async get_stock_detail({ stock_id }) {
      const id = cleanIds([stock_id], 1)[0];
      if (!id) return { error: "股票代號格式錯誤" };
      let d;
      try { d = await load(`stock/${id}.json`); } catch { return { error: `網站上沒有 ${id} 的資料（可能不是上市櫃普通股）` }; }
      return { result: JSON.parse(buildContext(d)), sources: [{ name: `${id} 個股資料`, path: `data/stock/${id}.json`, page: `#/stock/${id}` }] };
    },
    async compare_stocks({ stock_ids }) {
      const { u, t } = await uni(), ids = cleanIds(stock_ids, 6);
      if (ids.length < 2) return { error: "至少要兩檔有效的股票代號" };
      const rows = ids.map((id) => (t.at.has(id) ? t.row(t.at.get(id), COMPARE) : { stock_id: id, 說明: "網站資料裡沒有這檔" }));
      return { result: { ...dates(u), 股票: rows }, sources: [SRC_U] };
    },
    async screen_stocks({ filters = [], industry, sort_by = "score", order = "desc", limit = 10 }) {
      const { u, t } = await uni();
      const ok = new Set(u.fields);
      const fs = (Array.isArray(filters) ? filters : []).filter((f) => ok.has(f?.metric) && NUM_OPS[f?.op] && typeof f?.value === "number").slice(0, 8);
      const bad = (Array.isArray(filters) ? filters : []).filter((f) => !fs.includes(f)).map((f) => f?.metric);
      const sb = ok.has(sort_by) ? sort_by : "score", lim = Math.min(Math.max(+limit || 10, 1), 20);
      let idx = t.ids.map((_, i) => i).filter((i) => fs.every((f) => { const v = t.col(f.metric)[i]; return typeof v === "number" && NUM_OPS[f.op](v, f.value); }));
      if (industry) idx = idx.filter((i) => (t.col("industry")[i] || "").includes(industry));
      const n = idx.length, sv = t.col(sb), dir = order === "asc" ? 1 : -1;
      idx = idx.filter((i) => typeof sv[i] === "number").sort((a, b) => (sv[a] - sv[b]) * dir).slice(0, lim);
      const keys = [...new Set([...BRIEF, ...fs.map((f) => f.metric), sb])];
      return { result: { ...dates(u), 條件: fs, 產業: industry || "全部", 符合檔數: n, 排序: `${sb} ${order}`, 前幾名: idx.map((i) => t.row(i, keys)),
        ...(bad.length ? { 無法使用的條件: bad } : {}) }, sources: [SRC_U] };
    },
    async market_overview() {
      const { u, t } = await uni();
      const nm = (id) => (t.at.has(id) ? t.col("name")[t.at.get(id)] : "");
      const inds = [...(u.industries || [])].filter((x) => x.ret_1m != null).sort((a, b) => b.ret_1m - a.ret_1m);
      let events = [];
      try {
        const cal = await load("events/calendar.json"), today = (u.data_dates?.prices || "").slice(0, 10);
        const end = new Date(Date.parse(today) + 14 * 864e5).toISOString().slice(0, 10);
        events = (cal.events || []).filter((e) => e.event_date >= today && e.event_date <= end).slice(0, 30)
          .map((e) => ({ 日期: e.event_date, 股票: e.stock_id ? `${e.stock_id} ${e.name || nm(e.stock_id)}` : "所有公司", 事件: e.title, 狀態: e.status }));
      } catch { events = "事件資料暫時讀不到"; }
      const ch = u.changes;
      return { result: { ...dates(u), 大盤: u.market, 漲跌家數: u.breadth, 產業近一個月報酬_由高到低: inds.map((x) => ({ 產業: x.industry, 檔數: x.n, 當日: x.chg, 近一個月: x.ret_1m })),
        前20名變化: ch ? { 比較基準日: ch.since, 新進: ch.new.map((x) => `${x.stock_id} ${nm(x.stock_id)}（第 ${x.rank} 名）`), 掉出: ch.dropped.map((x) => `${x.stock_id} ${nm(x.stock_id)}（${x.why || ""}）`) } : null,
        未來兩週事件: events, 技術訊號檔數: t.col("signal").filter(Boolean).length },
      sources: [SRC_U, { name: "財報與事件日曆", path: "data/events/calendar.json", page: "#/calendar" }] };
    },
    async industry_report({ industry }) {
      const { u, t } = await uni(), q = String(industry || "").trim();
      const all = [...new Set(t.col("industry").filter(Boolean))];
      const name = all.find((x) => x === q) || all.find((x) => x.includes(q) || q.includes(x.replace(/業$/, "")));
      if (!name) return { error: `找不到產業「${q}」。可用的產業：${all.join("、")}` };
      const idx = t.ids.map((_, i) => i).filter((i) => t.col("industry")[i] === name);
      const sc = t.col("score");
      const top = [...idx].sort((a, b) => (sc[b] ?? -1) - (sc[a] ?? -1)).slice(0, 12).map((i) => t.row(i, BRIEF));
      const mc = t.col("market_cap"), big = [...idx].sort((a, b) => (mc[b] || 0) - (mc[a] || 0)).slice(0, 6).map((i) => `${t.ids[i]} ${t.col("name")[i]}`);
      const news = await newsFor({ industry: name, limit: 10 });
      return { result: { ...dates(u), 產業: name, 產業報酬: (u.industries || []).find((x) => x.industry === name) || null, 成分股數: idx.length,
        市值最大: big, 量化評分前段: top, 最近公告與新聞: news.items }, sources: [SRC_U, news.source] };
    },
    async search_news(input) {
      const news = await newsFor(input);
      return { result: news, sources: [news.source] };
    },
    async performance_summary() {
      const p = await load("performance/summary.json");
      const sc = p.scorecard || {}, pick = (x) => Object.fromEntries(Object.entries(x || {}).map(([h, v]) => [`第${h}個交易日出場`, v && {
        平均超額報酬: v.excess, 平均報酬: v.avg, 比全市場: v.vs_universe, 次數: v.cohorts, 贏過基準比例: v.beat_cohorts, 最差回撤: v.worst_mdd }]));
      return { result: { 資料日期: p.data_as_of, 定義: p.definitions, 警告: p.warnings, 紀錄: p.coverage,
        AI均衡型前20名: pick(sc.balanced), 技術訊號股: pick(sc.signals), 對照_全部可排名股票: pick(sc.universe) },
      sources: [{ name: "績效驗證", path: "data/performance/summary.json", page: "#/performance" }] };
    },
    async explain_metric({ query }) {
      const { u } = await uni(), q = String(query || "").trim();
      const hits = (u.catalog || []).filter((c) => c.key === q || c.name.includes(q) || q.includes(c.name)).slice(0, 5);
      return { result: hits.length ? hits : { 說明: "網站的指標資料庫裡沒有這個指標，可以用一般知識解釋，但要說明不是網站的定義" },
        sources: [{ name: "指標資料庫", path: "data/ai/universe.json", page: "#/catalog" }] };
    },
  };

  async function newsFor({ stock_ids, industry, keyword, kind = "all", limit = 10 } = {}) {
    const source = { name: "市場新聞中心", path: "data/news/latest.json", page: "#/news" };
    let N;
    try { N = await load("news/latest.json"); } catch { return { items: [], 說明: "新聞資料暫時讀不到", source }; }
    const ids = cleanIds(stock_ids, 10), lim = Math.min(Math.max(+limit || 10, 1), 20);
    let xs = N.items || [];
    if (ids.length) xs = xs.filter((x) => ids.includes(x.stock_id));
    if (industry) xs = xs.filter((x) => (x.industry || "").includes(industry));
    if (keyword) xs = xs.filter((x) => (x.title || "").includes(keyword) || (x.summary || "").includes(keyword) || (x.name || "").includes(keyword));
    if (kind === "announcement" || kind === "news") xs = xs.filter((x) => x.kind === kind);
    const seen = new Set(), out = [];
    for (const x of xs) {  // 同一事件的相似報導只留一篇
      const g = x.group || x.id;
      if (g && seen.has(g)) continue;
      seen.add(g);
      out.push({ 類型: x.kind === "announcement" ? "公司公告（官方）" : "財經新聞（媒體）", ...(x.kind === "announcement" ? { 重要性: ["", "一般", "值得注意", "重大"][x.importance || 1], 類別: x.category } : {}), 日期: x.date, 時間: x.time, 股票: x.stock_id ? `${x.stock_id} ${x.name || ""}` : null,
        產業: x.industry, 標題: x.title, 摘要: (x.summary || "").slice(0, 200) || undefined, 來源: x.source, 連結: x.link });
      if (out.length >= lim) break;
    }
    return { 涵蓋: N.coverage, 最後抓取: N.last_fetched_at, 限制: N.warnings, 符合筆數: xs.length, items: out, source };
  }

  return async (name, input) => {
    if (!run[name]) return { error: `沒有這個工具：${name}` };
    try {
      const r = await run[name](input && typeof input === "object" ? input : {});
      return r.error ? { error: r.error, sources: [] } : { result: r.result, sources: r.sources || [] };
    } catch (e) {
      return { error: `資料讀取失敗：${String(e?.message || e).slice(0, 120)}`, sources: [] };
    }
  };
}

// ---- 單一股票的資料包（從 data/stock/代號.json 精簡） ----
const ROW_KEYS = [
  "stock_id", "name", "industry", "market", "date", "close", "chg_pct", "market_cap",
  "per", "pbr", "dy", "per_pctile_1y", "eps_ttm", "eps_ttm_yoy", "eps_ttm_turn", "roe", "roa", "roe_years_15",
  "gross_margin", "op_margin", "debt_ratio", "ocf_to_ni", "fcf_yield", "fscore", "div_years", "payout",
  "rev_ym", "rev_yoy", "rev_yoy_3m", "rev_ytd_yoy", "fin_period", "val_date",
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
    "最近公告與新聞": (d.news || []).slice(0, 8).map((n) => ({ 類型: n.kind === "announcement" ? "公司公告" : "新聞", 日期: n.date, 標題: n.title, 來源: n.source, 連結: n.link })),
  };
  return JSON.stringify(ctx, (k, v) => (typeof v === "number" ? Math.round(v * 100) / 100 : v));
}
