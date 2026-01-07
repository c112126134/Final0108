# 日線策略設計文件 — 長短期均線 + RSI + 成交量 確認

## 目標
設計並實作一個以日線為頻率的交易策略，利用長短期均線判斷趨勢，並以 RSI 與成交量確認訊號，回測並輸出：

- MDD (最大回撤)
- Win rate (勝率)
- Profit Loss Ratio (盈虧比，平均賺 / 平均賠)

同時用 FastAPI 與 Jinja2 建立一個 HTML 頁面，顯示策略描述、回測績效報告與資金曲線圖（equity curve），目的為讓使用者知道何時應買進與賣出。

---

## 策略概念
1. 使用長短期簡單移動平均線 (SMA) 判斷中長期趨勢：
   - 多頭趨勢：SMA_short > SMA_long
   - 空頭趨勢：SMA_short < SMA_long

2. 在趨勢方向一致時，使用 RSI 作為動能確認（例如：RSI 升高表示動能增強）：
   - 在多頭趨勢時，若 RSI 突破某個門檻或呈現上升，就允許買進
   - 在空頭或 RSI 轉弱時賣出或空手

3. 用成交量做進一步確認：若當日成交量 > N 日均量 * multiplier（例如 1.2），代表訊號較可靠。

4. 進出場方式（簡單示範）：
   - 多單進場：當 SMA_short > SMA_long 且 RSI > rsi_buy_threshold（或 RSI 正在上升）且成交量符合條件 → 在當日收盤以可用資金買入整股數
   - 平倉：當 SMA_short < SMA_long 或 RSI 顯著回落 → 在當日收盤賣出全部持股

5. 風控與倉位：
   - 初始資金 (default) = 100000
   - 每次全部或使用全部現金買入（示範策略，方便解釋盈虧），可在參數中改為固定比例
   - 不使用槓桿與融資

---

## 指標與參數（可調）
- 短期 SMA: sma_short = 20
- 長期 SMA: sma_long = 50
- RSI 週期: rsi_period = 14
- RSI 進場門檻: rsi_buy = 55 （或 RSI 上升）
- RSI 出場門檻: rsi_buy = 45 （或 RSI 下降）
- 成交量均線: vol_ma = 20
- 成交量 multiplier: vol_mult = 1.2
- 初始資金: initial_capital = 100000

---

## 回測設計
1. 取資料來源：yfinance
2. 計算指標（SMA, RSI, volume MA）
3. 以日收盤價做為交易價格（簡化假設：以當日收盤價成交）
4. 追蹤每日淨值（equity），並記錄交易次數、每筆交易損益
5. 計算績效：
   - MDD：基於 equity curve 的最大回撤
   - Win rate：賺錢交易次數 / 交易次數
   - Profit Loss Ratio：平均賺 / 平均賠

---

## API 與 HTML 規格
- Single-file `final0108.py` 提供：
  - `/` : 顯示簡短策略描述、輸入表單（輸入 ticker, start, end, 參數）與最後一次回測結果（若有）
  - `/run?ticker=...&start=...&end=...` : 執行回測並回傳包含績效數值與資金曲線圖的 HTML

HTML 要包含：
- 策略描述（以文字說明）
- 回測績效報告（MDD, Win rate, Profit Loss Ratio, 總報酬）
- 資金曲線圖（內嵌 base64 圖片）

---

## 實作注意事項
- 單一 Python 檔案（`final0108.py`）: 內含資料抓取、策略邏輯、回測、圖表生成、FastAPI server 與 Jinja2 template
- 圖片以 matplotlib 產生並轉為 base64 內嵌於 HTML
- 提供 `--serve` 或直接在命令列執行 `python final0108.py` 可啟動一個測試回測並（選擇性）啟動 uvicorn server

---

## 結論
這個設計會提供一個能夠直觀了解買賣時點與整體績效的簡單框架。接下來我會把 `final0108.py` 實作出來，並用一個預設範例（例如 `AAPL`）做一次示範回測，驗證結果與指標計算皆正常。
