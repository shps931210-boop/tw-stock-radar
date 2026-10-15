-- 股栗子會員資料表：每位使用者一列，存觀察清單、自訂策略、個人偏好、研究筆記。
-- 在 Supabase 後台 SQL Editor 貼上整段、按 Run 執行一次即可（重複執行也不會出錯）。
-- Row Level Security：每個人只能讀寫自己的那一列，網站上的公開金鑰無法讀到別人的資料。

create table if not exists public.user_state (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  watch      jsonb not null default '{}'::jsonb,   -- 觀察清單 {代號: {added, price}}
  strategies jsonb not null default '[]'::jsonb,   -- 自訂條件選股策略
  pref       jsonb not null default '{}'::jsonb,   -- 我的 AI 排行偏好（期間、風格、風險、隱藏的股票）
  notes      jsonb not null default '{}'::jsonb,   -- 個股研究筆記 {代號: 文字}
  updated_at timestamptz not null default now()
);
-- 版本號：每次存檔 +1。網站存檔時要帶上次讀到的版本號，不一樣就代表另一台裝置剛改過，不覆寫（避免資料遺失）。
alter table public.user_state add column if not exists version integer not null default 1;
-- 我的投資組合（V21）：交易紀錄 {trades: [...], settings: {...}}，實際持股與模擬投資分開記錄。
alter table public.user_state add column if not exists portfolio jsonb not null default '{}'::jsonb;

alter table public.user_state enable row level security;

drop policy if exists "own row: read" on public.user_state;
drop policy if exists "own row: insert" on public.user_state;
drop policy if exists "own row: update" on public.user_state;
drop policy if exists "own row: delete" on public.user_state;
create policy "own row: read"   on public.user_state for select to authenticated using ((select auth.uid()) = user_id);
create policy "own row: insert" on public.user_state for insert to authenticated with check ((select auth.uid()) = user_id);
create policy "own row: update" on public.user_state for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
create policy "own row: delete" on public.user_state for delete to authenticated using ((select auth.uid()) = user_id);

grant usage on schema public to authenticated;
grant select, insert, update, delete on public.user_state to authenticated;
revoke all on public.user_state from anon;

-- ===== AI 問答用量（每位會員每天上限、全站每天上限） =====
-- 只有 AI 問答後端會呼叫 use_ai_quota()，並帶上會員的登入憑證；沒登入的人呼叫會得到 false。
-- 上限寫死在函式裡，使用者不能自己改；要調整請改下面兩個數字後重新執行整段 SQL。
create table if not exists public.ai_usage (
  day     date not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  n       integer not null default 0,
  primary key (day, user_id)
);
-- 實際花費（美元）：AI 問答後端回答完後依 Token 用量回報，下面的上限會一起檢查。
alter table public.ai_usage add column if not exists cost_usd numeric(10, 4) not null default 0;
alter table public.ai_usage enable row level security;
drop policy if exists "own usage: read" on public.ai_usage;
create policy "own usage: read" on public.ai_usage for select to authenticated using ((select auth.uid()) = user_id);
grant select on public.ai_usage to authenticated;
revoke all on public.ai_usage from anon;

create or replace function public.use_ai_quota()
returns json
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := auth.uid();
  today date := (now() at time zone 'Asia/Taipei')::date;
  per_user constant integer := 20;    -- 每位會員每天最多 20 題
  per_site constant integer := 100;   -- 全站每天最多 100 題
  usd_user constant numeric := 0.50;  -- 每位會員每天最多花 0.5 美元
  usd_site constant numeric := 3.00;  -- 全站每天最多花 3 美元（另外請在 Anthropic Console 設定每月支出上限）
  mine integer;
  total integer;
  cost_mine numeric;
  cost_total numeric;
begin
  if uid is null then
    return json_build_object('ok', false, 'reason', 'login');
  end if;
  perform pg_advisory_xact_lock(hashtext('ai_usage'));  -- 同時有很多人問時，計數不會算錯
  select coalesce(sum(n), 0), coalesce(sum(cost_usd), 0) into total, cost_total from public.ai_usage where day = today;
  select coalesce(n, 0), coalesce(cost_usd, 0) into mine, cost_mine from public.ai_usage where day = today and user_id = uid;
  mine := coalesce(mine, 0);
  cost_mine := coalesce(cost_mine, 0);
  if total >= per_site or cost_total >= usd_site then
    return json_build_object('ok', false, 'reason', 'site', 'used', mine, 'limit', per_user);
  end if;
  if mine >= per_user or cost_mine >= usd_user then
    return json_build_object('ok', false, 'reason', 'user', 'used', mine, 'limit', per_user);
  end if;
  insert into public.ai_usage (day, user_id, n) values (today, uid, 1)
    on conflict (day, user_id) do update set n = public.ai_usage.n + 1;
  return json_build_object('ok', true, 'used', mine + 1, 'limit', per_user);
end;
$$;
revoke all on function public.use_ai_quota() from public, anon;
grant execute on function public.use_ai_quota() to authenticated;

-- AI 問答回答完後回報這一題的花費（依 Token 用量計算）。單次最多記 0.5 美元，負數當 0。
create or replace function public.record_ai_cost(p_cost numeric)
returns json
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := auth.uid();
  today date := (now() at time zone 'Asia/Taipei')::date;
  c numeric := least(greatest(coalesce(p_cost, 0), 0), 0.5);
begin
  if uid is null then
    return json_build_object('ok', false, 'reason', 'login');
  end if;
  insert into public.ai_usage (day, user_id, n, cost_usd) values (today, uid, 0, c)
    on conflict (day, user_id) do update set cost_usd = public.ai_usage.cost_usd + c;
  return json_build_object('ok', true);
end;
$$;
revoke all on function public.record_ai_cost(numeric) from public, anon;
grant execute on function public.record_ai_cost(numeric) to authenticated;

-- ===== 智慧警報（站內通知）=====
-- 規則、通知、比較基準都只屬於自己的帳號。網站只用公開金鑰；通知由後端排程產生（見下方權限）。
-- 同一則通知有固定代號 event_key（規則:股票:資料日），unique (user_id, event_key) 保證不會重複通知。
create table if not exists public.alert_rules (
  user_id    uuid not null references auth.users (id) on delete cascade,
  rule_type  text not null check (rule_type in ('signal', 'top', 'score', 'risk', 'exdiv')),
  enabled    boolean not null default true,
  params     jsonb not null default '{}'::jsonb check (pg_column_size(params) < 2000),
  updated_at timestamptz not null default now(),
  primary key (user_id, rule_type)
);
create table if not exists public.user_notifications (
  id         bigint generated always as identity primary key,
  user_id    uuid not null references auth.users (id) on delete cascade,
  event_key  text not null check (char_length(event_key) <= 120),
  rule_type  text not null check (char_length(rule_type) <= 20),
  stock_id   text check (char_length(stock_id) <= 10),
  title      text not null check (char_length(title) <= 200),
  body       text check (char_length(body) <= 1000),
  data_date  date,
  created_at timestamptz not null default now(),
  read_at    timestamptz,
  unique (user_id, event_key)
);
create index if not exists user_notifications_recent on public.user_notifications (user_id, created_at desc);
create table if not exists public.alert_processing_state (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  data_date  date,                                   -- 上次比較用的資料日
  baseline   jsonb not null default '{}'::jsonb check (pg_column_size(baseline) < 200000),  -- 觀察清單股票上次的分數、名次、門檻
  updated_at timestamptz not null default now()
);

alter table public.alert_rules enable row level security;
alter table public.user_notifications enable row level security;
alter table public.alert_processing_state enable row level security;
do $$
declare t text;
begin
  foreach t in array array['alert_rules', 'user_notifications', 'alert_processing_state'] loop
    execute format('drop policy if exists "own rows: read" on public.%I', t);
    execute format('drop policy if exists "own rows: insert" on public.%I', t);
    execute format('drop policy if exists "own rows: update" on public.%I', t);
    execute format('drop policy if exists "own rows: delete" on public.%I', t);
    execute format('create policy "own rows: read" on public.%I for select to authenticated using ((select auth.uid()) = user_id)', t);
    execute format('create policy "own rows: insert" on public.%I for insert to authenticated with check ((select auth.uid()) = user_id)', t);
    execute format('create policy "own rows: update" on public.%I for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id)', t);
    execute format('create policy "own rows: delete" on public.%I for delete to authenticated using ((select auth.uid()) = user_id)', t);
    execute format('revoke all on public.%I from anon', t);
  end loop;
end $$;
-- 通知規則：會員自己設定。
grant select, insert, update, delete on public.alert_rules to authenticated;
-- 正式通知與處理進度：只由後端（GitHub Actions 的 app/alerts.py，使用 secret key）寫入。
-- 會員只能讀自己的通知、標為已讀；不能自己新增、刪除通知，也不能改處理進度。
revoke insert, update, delete on public.user_notifications from authenticated;
grant select on public.user_notifications to authenticated;
grant update (read_at) on public.user_notifications to authenticated;
revoke insert, update, delete on public.alert_processing_state from authenticated;
grant select on public.alert_processing_state to authenticated;
-- 後端（secret key = service_role）需要明確的權限；新的 Supabase 專案不會自動給。
grant usage on schema public to service_role;
grant select on public.user_state to service_role;
grant select on public.alert_rules to service_role;
grant select, insert, update on public.user_notifications to service_role;
grant select, insert, update on public.alert_processing_state to service_role;

-- 每人最多保留 500 則通知，超過就刪掉最舊的（避免免費資料庫被塞滿）
create or replace function public.trim_notifications()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  delete from public.user_notifications
   where user_id = new.user_id
     and id in (select id from public.user_notifications where user_id = new.user_id order by created_at desc offset 500);
  return null;
end;
$$;
drop trigger if exists trim_notifications on public.user_notifications;
create trigger trim_notifications after insert on public.user_notifications for each row execute function public.trim_notifications();

-- ===== 方案（先預留，不收費）=====
-- 目前每個人都是 free。只能讀自己的方案，網站不能自己改成 pro。
create table if not exists public.user_plan (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  plan       text not null default 'free' check (plan in ('free', 'pro')),
  updated_at timestamptz not null default now()
);
alter table public.user_plan enable row level security;
drop policy if exists "own plan: read" on public.user_plan;
create policy "own plan: read" on public.user_plan for select to authenticated using ((select auth.uid()) = user_id);
grant select on public.user_plan to authenticated;
revoke all on public.user_plan from anon;
revoke insert, update, delete on public.user_plan from authenticated;

-- ===== 刪除帳號（會員自己按「刪除帳號」時呼叫）=====
-- 刪除登入帳號；觀察清單、通知、規則、用量紀錄都設定了 on delete cascade，會一起刪掉。無法復原。
create or replace function public.delete_my_account()
returns json
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := auth.uid();
begin
  if uid is null then
    return json_build_object('ok', false, 'reason', 'login');
  end if;
  delete from auth.users where id = uid;
  return json_build_object('ok', true);
end;
$$;
revoke all on function public.delete_my_account() from public, anon;
grant execute on function public.delete_my_account() to authenticated;

-- ===== 優先補抓財報（V21.1）=====
-- 會員在個股頁按「申請優先補抓」：夜間補抓（GitHub Actions，secret key）會先處理這些股票，補完就刪除請求。
-- 每人同時最多 20 檔；只能看到、刪除自己的請求。
create table if not exists public.fin_requests (
  user_id      uuid not null default auth.uid() references auth.users (id) on delete cascade,
  stock_id     text not null check (stock_id ~ '^[0-9]{4,6}[A-Z]?$'),
  requested_at timestamptz not null default now(),
  primary key (user_id, stock_id)
);
alter table public.fin_requests enable row level security;
drop policy if exists "own requests: read" on public.fin_requests;
drop policy if exists "own requests: insert" on public.fin_requests;
drop policy if exists "own requests: delete" on public.fin_requests;
create policy "own requests: read" on public.fin_requests for select to authenticated using ((select auth.uid()) = user_id);
create policy "own requests: insert" on public.fin_requests for insert to authenticated
  with check ((select auth.uid()) = user_id and (select count(*) from public.fin_requests r where r.user_id = (select auth.uid())) < 20);
create policy "own requests: delete" on public.fin_requests for delete to authenticated using ((select auth.uid()) = user_id);
grant select, insert, delete on public.fin_requests to authenticated;
revoke all on public.fin_requests from anon;
grant select, delete on public.fin_requests to service_role;
