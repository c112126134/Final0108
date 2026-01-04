# 期末專案
---
## 組員
C112126107 莊品柔
C112126134 吳宗叡

---
## 環境設定說明
- Python 版本：3.11.5
- 使用 FastAPI 建立Web伺服器
- 使用 Jinja2 建立HTML頁面
- 使用 Github Copilot 執行程式＆網頁設計
- 啟動網頁前需執行：`python.exe Final0108\\final0108.py --serve`
- 網頁位址：http://127.0.0.1:8000/

---
## 執行方法


---
## 專案設計說明

#### 我們這組的策略為：
1. 透過長短期均線（SMA）判斷中長期趨勢
2. 相對強弱指標（RSI）突破時買進，轉弱時賣出
3. 最後用成交量做進一步確認

  * **進場時機**
    短SMA > 長SMA （黃金交叉），且RSI > 進場門檻，或RSI正在上升

  * **平倉時機**
短SMA < 長SMA （死亡交叉），且RSI存在顯著回落

#### 專案架構：
- `scrpts - render_index.py`：輔助腳本，用於生成靜態索引或報表
- `static - style.css`：網頁頁面的CSS設計
- `README.MD`：專案摘要與使用說明
- `final0108.csv`：回測的輸出資料
- `final0108.py`：主程式，完整流程為：__依賴檢查 → 下載資料 → 計算指標 → 產生訊號 → 回測 → 產生報表與 FastAPI 網頁/API__
- `final0108_design.md`：專案設計說明
- `pyproject.html`：Python 專案元資料
- `requirements.txt`：相依套件清單（pip install 可用）
  
---
## 心得總結
