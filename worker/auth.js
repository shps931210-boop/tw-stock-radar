// 會員驗證與每日用量：用使用者的 Supabase 登入憑證呼叫資料庫函式 use_ai_quota()（supabase/schema.sql）。
// 憑證無效、沒登入、超過每人或全站每日上限，都不會呼叫付費 AI。任何檢查失敗一律拒絕（寧可不回答，也不要多花錢）。

const MSG = {
  login: "請先登入才能使用 AI 問答（沒登入可以用免費的自動問答）",
  user: (q) => `今天的 ${q.limit} 題已經用完了，明天再來（免費的自動問答仍可使用）`,
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
