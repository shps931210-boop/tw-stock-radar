/* 股栗子｜我的投資組合：交易紀錄、持股損益、報酬率的計算（純函式，網站與測試共用）。
 *
 * 規則：
 * - 成本用「移動平均成本法」：買進成本含手續費；賣出時按平均成本沖銷，已實現損益 = 賣出金額 − 手續費 − 交易稅 − 沖銷成本。
 * - 手續費 = 成交金額 × 0.1425% × 折扣，無條件捨去；低於最低手續費（整股預設 20 元、零股預設 1 元，可自行設定）以最低計。
 * - 交易稅（賣出才有）：股票 0.3%、ETF 0.1%，無條件捨去。手續費和交易稅都可以手動改成券商實收的金額。
 * - 現金股利另外記一筆「股利」，算在報酬裡；沒有記錄的股利不會自動加入（除息日股價下跌會讓帳面報酬偏低）。
 * - 時間加權報酬（TWR）：每天「（收盤市值 ＋ 當天賣出收入 ＋ 股利）÷（前一天市值 ＋ 當天買進金額）」連乘，不受進出金額大小影響，
 *   適合和 0050 比較選股與操作能力；資金加權報酬（MWR，年化 XIRR）：把每筆進出當成現金流，反映你實際的投資結果。
 * - 0050 對照：同一天用同樣的金額買賣 0050（還原除權息），看現在會是多少錢。
 */
(function (root) {
  const FEE_RATE = 0.001425;
  const DEFAULTS = { discount: 1, min_fee: 20, odd_min_fee: 1, tax_stock: 0.003, tax_etf: 0.001 };
  const settings = (p) => ({ ...DEFAULTS, ...((p && p.settings) || {}) });
  const isEtf = (sid) => /^00/.test(String(sid));

  function estFee(amount, shares, st) {
    if (!(amount > 0)) return 0;
    const f = Math.floor(amount * FEE_RATE * st.discount);
    return Math.max(shares >= 1000 ? st.min_fee : st.odd_min_fee, f);
  }
  function estTax(amount, sid, st) {
    return amount > 0 ? Math.floor(amount * (isEtf(sid) ? st.tax_etf : st.tax_stock)) : 0;
  }

  const live = (trades, book) => (trades || []).filter((t) => !t.del && (!book || (t.book || "real") === book))
    .slice().sort((a, b) => (a.date + (a.at || "")).localeCompare(b.date + (b.at || "")));

  // 逐筆計算持股；errors 會列出「賣出超過持股」等問題（依日期順序檢查，補登過去的交易也會檢查）
  function ledger(trades, book) {
    const pos = {}, errors = [];
    for (const t of live(trades, book)) {
      const p = pos[t.sid] || (pos[t.sid] = { sid: t.sid, shares: 0, cost: 0, realized: 0, divs: 0, fees: 0, taxes: 0, bought: 0, first: t.date, last: t.date, n: 0 });
      p.last = t.date; p.n += 1;
      if (t.side === "buy") {
        const amt = t.shares * t.price + (t.fee || 0);
        p.shares += t.shares; p.cost += amt; p.fees += t.fee || 0; p.bought += amt;
        if (p.shares === t.shares) p.open = t.date;
      } else if (t.side === "sell") {
        if (t.shares > p.shares + 1e-9) { errors.push({ id: t.id, sid: t.sid, date: t.date, msg: `${t.date} 賣出 ${t.shares} 股，但當時只持有 ${p.shares} 股` }); continue; }
        const out = p.shares ? p.cost * t.shares / p.shares : 0;
        p.realized += t.shares * t.price - (t.fee || 0) - (t.tax || 0) - out;
        p.cost -= out; p.shares -= t.shares; p.fees += t.fee || 0; p.taxes += t.tax || 0;
        if (p.shares < 1e-9) { p.shares = 0; p.cost = 0; }
      } else if (t.side === "div") {
        p.divs += t.amount || 0;
      }
    }
    return { pos, errors };
  }

  // 新增或修改一筆之前先檢查：數字合理、日期不在未來、賣出不超過當時持股（之後的賣出也不能因此變成超賣）
  function validate(trades, t, today) {
    const e = [];
    if (!t.sid) e.push("請選擇股票");
    if (!["buy", "sell", "div"].includes(t.side)) e.push("請選擇買進、賣出或股利");
    if (!t.date || !/^\d{4}-\d{2}-\d{2}$/.test(t.date)) e.push("請輸入日期");
    else if (today && t.date > today) e.push("日期不能在未來");
    if (t.side === "div") { if (!(t.amount > 0)) e.push("請輸入股利金額"); }
    else {
      if (!(Number.isInteger(t.shares) && t.shares > 0)) e.push("股數要是正整數");
      if (!(t.price > 0)) e.push("請輸入實際成交價");
      if ((t.fee || 0) < 0 || (t.tax || 0) < 0) e.push("手續費和交易稅不能是負數");
    }
    if (!e.length) {
      const others = (trades || []).filter((x) => x.id !== t.id);
      const before = ledger(others, t.book || "real").errors.length;
      const r = ledger([...others, t], t.book || "real");
      if (r.errors.length > before) e.push(r.errors.find((x) => x.sid === t.sid)?.msg || r.errors.at(-1).msg);
    }
    return e;
  }

  // 目前持股：prices = {代號: 最新收盤價}
  function holdings(L, prices, st) {
    const rows = [];
    for (const p of Object.values(L.pos)) {
      const px = prices[p.sid];
      const mv = p.shares && px != null ? p.shares * px : 0;
      const sellCost = p.shares && px != null ? estFee(mv, p.shares, st) + estTax(mv, p.sid, st) : 0;
      rows.push({ ...p, price: px ?? null, mv, avg: p.shares ? p.cost / p.shares : null,
        unreal: p.shares && px != null ? mv - p.cost : null, unreal_net: p.shares && px != null ? mv - p.cost - sellCost : null,
        unreal_pct: p.shares && px != null && p.cost ? (mv / p.cost - 1) * 100 : null,
        total: (p.shares && px != null ? mv - p.cost : 0) + p.realized + p.divs });
    }
    const mvAll = rows.reduce((a, r) => a + r.mv, 0);
    rows.forEach((r) => (r.weight = mvAll ? r.mv / mvAll * 100 : 0));
    return rows.sort((a, b) => b.mv - a.mv || b.total - a.total);
  }

  // 現金流（投資人角度：買進為負、賣出與股利為正），最後一筆是目前市值
  function cashflows(trades, book, mvNow, asof) {
    const f = [];
    for (const t of live(trades, book)) {
      if (t.side === "buy") f.push([t.date, -(t.shares * t.price + (t.fee || 0))]);
      else if (t.side === "sell") f.push([t.date, t.shares * t.price - (t.fee || 0) - (t.tax || 0)]);
      else if (t.side === "div") f.push([t.date, t.amount || 0]);
    }
    if (mvNow > 0) f.push([asof, mvNow]);
    return f;
  }

  const days = (a, b) => (Date.parse(b) - Date.parse(a)) / 864e5;
  // 年化資金加權報酬（XIRR）；期間太短（< 30 天）年化沒有意義，回傳 null
  function xirr(flows) {
    if (flows.length < 2 || !flows.some((x) => x[1] > 0) || !flows.some((x) => x[1] < 0)) return null;
    const t0 = flows[0][0];
    if (days(t0, flows.at(-1)[0]) < 30) return null;
    const npv = (r) => flows.reduce((a, [d, v]) => a + v / Math.pow(1 + r, days(t0, d) / 365), 0);
    let lo = -0.9999, hi = 10;
    if (npv(lo) * npv(hi) > 0) return null;
    for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (npv(lo) * npv(m) <= 0) hi = m; else lo = m; }
    return (lo + hi) / 2;
  }

  // 每日市值與時間加權報酬。series = {代號: [[日期, 收盤], ...]}（未還原的實際股價）
  function daily(trades, book, series, asof) {
    const T = live(trades, book);
    if (!T.length) return { curve: [], twr: null };
    const start = T[0].date;
    const dates = [...new Set(Object.values(series).flatMap((s) => s.map((x) => x[0])))].filter((d) => d >= start && (!asof || d <= asof)).sort();
    const close = {}, last = {};
    for (const [sid, s] of Object.entries(series)) close[sid] = Object.fromEntries(s);
    const sh = {}, byDay = {};
    for (const t of T) (byDay[t.date] = byDay[t.date] || []).push(t);
    // 交易日之前（例如週末補登）的交易併到下一個交易日
    const pend = Object.keys(byDay).filter((d) => !dates.includes(d)).sort();
    let prev = 0, g = 1, k = 0;
    const curve = [];
    for (const d of dates) {
      let inflow = 0, outflow = 0;
      const todays = [];
      while (k < pend.length && pend[k] <= d) { todays.push(...byDay[pend[k]]); k++; }
      todays.push(...(byDay[d] || []));
      for (const t of todays) {
        if (t.side === "buy") { sh[t.sid] = (sh[t.sid] || 0) + t.shares; inflow += t.shares * t.price + (t.fee || 0); last[t.sid] = last[t.sid] ?? t.price; }
        else if (t.side === "sell") { sh[t.sid] = (sh[t.sid] || 0) - t.shares; outflow += t.shares * t.price - (t.fee || 0) - (t.tax || 0); }
        else if (t.side === "div") outflow += t.amount || 0;
      }
      let v = 0;
      for (const [sid, n] of Object.entries(sh)) {
        if (n <= 0) continue;
        const c = close[sid]?.[d];
        if (c != null) last[sid] = c;
        v += n * (last[sid] ?? 0);
      }
      const base = prev + inflow;
      if (base > 0) g *= (v + outflow) / base;
      curve.push([d, v, g]);
      prev = v;
    }
    return { curve, twr: curve.length ? (g - 1) * 100 : null };
  }

  // 0050 對照：同一天用同樣的金額買進／賣出 0050（還原股價），現在會值多少；也算同期間 0050 的報酬
  function shadow(trades, book, bench, asof) {
    const T = live(trades, book);
    if (!T.length || !bench.length) return null;
    const px = (d) => { let v = null; for (const [bd, c] of bench) { if (bd <= d) v = c; else break; } return v; };
    let units = 0, short = false;
    for (const t of T) {
      const p = px(t.date); if (!p) continue;
      if (t.side === "buy") units += (t.shares * t.price + (t.fee || 0)) / p;
      else if (t.side === "sell") units -= (t.shares * t.price - (t.fee || 0) - (t.tax || 0)) / p;
      else if (t.side === "div") units -= (t.amount || 0) / p;
      if (units < 0) { short = true; units = 0; }
    }
    const p0 = px(T[0].date), p1 = px(asof || bench.at(-1)[0]);
    return { value: units * (p1 || 0), bench_ret: p0 && p1 ? (p1 / p0 - 1) * 100 : null, capped: short };
  }

  const PF = { DEFAULTS, settings, estFee, estTax, isEtf, live, ledger, validate, holdings, cashflows, xirr, daily, shadow };
  if (typeof module !== "undefined" && module.exports) module.exports = PF;
  else root.PF = PF;
})(typeof window !== "undefined" ? window : globalThis);
