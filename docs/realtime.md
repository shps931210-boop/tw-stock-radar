# 股栗子盤中行情架構（V12）

目前狀態：**架構與模擬行情完成，還沒有接上任何真實盤中行情**。網站仍然只顯示盤後正式資料。

## 架構

```
盤中（暫時資料）                                   盤後（正式資料，原本的流程不變）
行情供應商（元大 SPARK／模擬／盤後收盤）           證交所、櫃買中心、FinMind
        ↓ app/realtime/providers.py                       ↓ GitHub Actions（app/update.py）
行情接收程式 app/realtime/receiver.py               資料庫 data/stocks.db（prices 表）
（常駐電腦：訂閱、心跳、重連、去重、限額）                 ↓
        ↓                                           AI 預測封存、到期對帳、每日學習、模型比賽
data/intraday/quotes.json（app/realtime/store.py）           ↓
        ↓（未來：Cloudflare Worker 讀取、快取）       GitHub Pages 網站
網站盤中價格（未完成；過期或斷線時退回盤後價格）
```

- 盤中資料只寫 `data/intraday/quotes.json`，不會寫進資料庫；AI 預測、封存、對帳、訓練只讀資料庫的正式收盤價（測試 `tests/test_realtime.py` 檢查）。
- 每筆行情都帶 `kind`：`intraday`（盤中）、`simulated`（模擬，不是真實行情）、`official`（盤後）。
- 盤中超過 2 分鐘沒有新行情、或接收程式未連線，就退回最後正式收盤價，並標示價格時間。

## 試跑（模擬行情）

```
python -m app.realtime.receiver --provider simulated --ids 2330,0050 --seconds 30
```

## 接上元大 SPARK API 的步驟

1. 向元大營業員確認：行情是否另外收費；**能不能在公開網站展示、轉發給其他使用者**；訂閱檔數與連線限制；網站將來收費時是否需要商業授權。沒有確認可公開展示前，行情只能自己看。
2. 完成元大的申請流程（風險預告書、API 測試、營業員開通）。
3. 在盤中會一直開著的電腦安裝元大官方 SDK，帳號與憑證只放在這台電腦的環境變數：
   `YUANTA_ID`、`YUANTA_PASSWORD`、`YUANTA_CERT_PATH`、`YUANTA_CERT_PASSWORD`（不能放 GitHub、不能放網站、不要貼在聊天室）。
4. 確認授權後再設定 `YUANTA_DISPLAY_LICENSED=1`；沒有這個設定，接收程式會拒絕啟動。
5. 依元大官方文件完成 `YuantaSparkProvider.connect／subscribe／poll`（目前是空的接口），先用台積電、0050 幾檔測試。
6. 再做網站顯示：由一個後端（例如 Cloudflare Worker 讀快取）提供價格，網站不直接連元大、不帶任何帳號。只做行情，不做下單。

## 限制

- GitHub Pages、GitHub Actions 不能常駐接收行情；盤中更新要靠一台開著的電腦（電腦關機就停止更新，網站會自動退回盤後價格）。
- Cloudflare Workers 免費方案適合做快取與 API，不適合執行元大的 Python SDK。
