// 會員驗證與每日用量：用使用者的 Supabase 登入憑證呼叫資料庫函式 use_ai_quota()（supabase/schema.sql）。
// 憑證無效、沒登入、超過每人或全站每日上限，都不會呼叫付費 AI。任何檢查失敗一律拒絕（寧可不回答，也不要多花錢）。

const MSG = {
  login: "請先登入才能使用 AI 問答（沒登入可以用免費的自動問答）",
  user: (q) => `你今天的 AI 問答額度已經用完了（每人每天 ${q.limit || 20} 題，或約 0.5 美元的用量），明天再來（免費的自動問答仍可使用）`,
  site: "今天全站的 AI 問答額度已經用完，明天再來（免費的自動問答仍可使用）",
};

export async function checkQuota(env, authHeader, f = fetch) {
  const token = String(authHeader || "").replace(/^Bearer\s+/i, "").trim();
  if (!env.SUPABASE_URL || !env.SUPABASE_KEY) return { ok: false, status: 503, error: "AI 問答還沒接上會員系統" };
  if (!token) return { ok: false, status: 401, error: MSG.login };
  let r;
  try {
    r = await f(`${env.SUPABASE_URL}/rest/v1/rpc/use_ai_quota`, {
      method: "POST",
      headers: { apikey: env.SUPABASE_KEY, Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: "{}",
    });
  } catch {
    return { ok: false, status: 503, error: "暫時無法確認使用額度，請稍後再試" };
  }
  if (r.status === 401 || r.status === 403) return { ok: false, status: 401, error: "登入已過期，請重新登入" };
  if (!r.ok) return { ok: false, status: 503, error: "暫時無法確認使用額度，請稍後再試" };
  const q = await r.json().catch(() => ({}));
  if (q.ok === true) return { ok: true, used: q.used, limit: q.limit };
  const reason = q.reason in MSG ? q.reason : "login";
  const msg = typeof MSG[reason] === "function" ? MSG[reason](q) : MSG[reason];
  return { ok: false, status: reason === "login" ? 401 : 429, error: msg, reason };
}

// 各模型每百萬 Token 的價格（美元）。快取寫入以輸入價 1.25 倍計（5 分鐘快取）。不認得的模型用最貴的 Opus 價格估算，寧可高估。
export const PRICES = {
  "claude-opus-5-5": { in: 4, out: 20, cacheRead: 0.2 },
  "claude-haiku-5-5": { in: 0.1, out: 0.5, cacheRead: 0.01 },
};
export function costOf(model, usage) {
  const key = Object.keys(PRICES).find((k) => String(model || "").startsWith(k));
  const p = PRICES[key] || PRICES["claude-opus-5-5"], u = usage || {};
  const usd = ((u.input_tokens || 0) * p.in + (u.cache_creation_input_tokens || 0) * p.in * 1.25
    + (u.cache_read_input_tokens || 0) * p.cacheRead + (u.output_tokens || 0) * p.out) / 1e6;
  return Math.round(usd * 1e6) / 1e6;
}
/** 回答完回報花費。失敗只記錄，不影響已經回答的內容；下一題的額度檢查會看到累計花費。 */
export async function recordCost(env, authHeader, usd, f = fetch) {
  const token = String(authHeader || "").replace(/^Bearer\s+/i, "").trim();
  if (!env.SUPABASE_URL || !env.SUPABASE_KEY || !token) return false;
  try {
    const r = await f(`${env.SUPABASE_URL}/rest/v1/rpc/record_ai_cost`, {
      method: "POST",
      headers: { apikey: env.SUPABASE_KEY, Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ p_cost: usd }),
    });
    return r.ok;
  } catch {
    return false;
  }
}
