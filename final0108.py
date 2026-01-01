"""
final0108.py

單一檔案示範：
- 從 yfinance 下載日線資料
- 計算 SMA (short/long), RSI, volume MA
- 使用長短期均線判斷趨勢，並以 RSI + 成交量確認進場
- 執行簡單的逐日回測（以收盤價成交）
- 輸出 MDD, Win rate, Profit/Loss Ratio
- 產生資金曲線圖並用 FastAPI + Jinja2 呈現於 HTML（base64 圖片嵌入）

使用：
python final0108.py            # 執行範例回測並輸出結果
python final0108.py --serve   # 啟動 web server (uvicorn)

Dependencies: yfinance, pandas, numpy, matplotlib, fastapi, uvicorn
"""

# ---------------------------
# Dependency checks (provide helpful message and optional auto-install)
# ---------------------------
import sys
import subprocess
import os

required = ['fastapi', 'yfinance', 'pandas', 'numpy', 'matplotlib', 'jinja2']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except Exception:
        missing.append(pkg)

# stdlib
import io
import base64
import argparse
import datetime

# If missing packages, optionally auto-install when --install-deps flag is provided
install_flag = ('--install-deps' in sys.argv) or (os.environ.get('AUTO_INSTALL_DEPS') == '1')
if missing:
    msg = "Missing required Python packages: " + ", ".join(missing)
    if install_flag:
        print('Missing packages detected; attempting to install:', missing)
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        except subprocess.CalledProcessError as e:
            # pip may be missing in this interpreter; try bootstrapping via ensurepip
            print('pip not available; attempting to bootstrap pip with ensurepip...')
            try:
                subprocess.check_call([sys.executable, '-m', 'ensurepip', '--upgrade'])
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            except Exception as e2:
                raise SystemExit(msg + '\nAuto-install failed: ' + str(e2))
        # try importing again
        failed = []
        for pkg in missing:
            try:
                __import__(pkg)
            except Exception:
                failed.append(pkg)
        if failed:
            raise SystemExit(msg + '\nSome packages still failed to import after install: ' + ', '.join(failed))
        else:
            print('Successfully installed missing packages; continuing.')
            # remove --install-deps from argv so argparse in main() doesn't error
            if '--install-deps' in sys.argv:
                sys.argv = [a for a in sys.argv if a != '--install-deps']
            # proceed to import specific names below
    else:
        raise SystemExit(msg + ".\nInstall with: pip install " + " ".join(missing) + "\nOr: pip install -r requirements.txt\nTo auto-install, run: python final0108.py --install-deps")

# Import specific names we use in the rest of the script
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
# use non-interactive backend to avoid GUI/tkinter issues when running in tests or servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jinja2 import Template
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# ---------------------------
# 指標函數
# ---------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# ---------------------------
# 產生訊號
# ---------------------------

def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    df['sma_short'] = sma(df['Adj Close'], params['sma_short'])
    df['sma_long'] = sma(df['Adj Close'], params['sma_long'])
    df['rsi'] = rsi(df['Adj Close'], params['rsi_period'])
    df['vol_ma'] = df['Volume'].rolling(params['vol_ma']).mean()

    # 趨勢判斷
    df['trend_long'] = np.where(df['sma_short'] > df['sma_long'], 1, -1)

    # RSI 上升判斷（與前一日比較）
    df['rsi_up'] = df['rsi'] > df['rsi'].shift(1)

    # 成交量確認 (robust to odd shapes)
    vol_arr = np.asarray(df['Volume'])
    vol_ma_arr = np.asarray(df['vol_ma'])
    # squeeze any extra dimensions
    if vol_arr.ndim > 1:
        vol_arr = vol_arr.squeeze()
    if vol_ma_arr.ndim > 1:
        vol_ma_arr = vol_ma_arr.squeeze()
    # ensure 1D
    try:
        vol_ok_arr = vol_arr > (vol_ma_arr * params['vol_mult'])
        # if result shape mismatch, fallback to elementwise by index
        if vol_ok_arr.shape[0] != len(df):
            raise ValueError('shape mismatch')
        df['vol_ok'] = vol_ok_arr
    except Exception:
        # fallback: conservative default (no volume confirmation)
        df['vol_ok'] = False

    # 進出場
    df['signal'] = 0
    # buy conditions
    buy_cond = (
        (df['trend_long'] == 1) &
        (df['rsi'] > params['rsi_buy']) &
        (df['rsi_up']) &
        (df['vol_ok'])
    )
    df.loc[buy_cond, 'signal'] = 1

    # sell conditions
    sell_cond = (
        (df['trend_long'] == -1) |
        (df['rsi'] < params['rsi_sell'])
    )
    df.loc[sell_cond, 'signal'] = -1

    return df

# ---------------------------
# 回測引擎（簡單範例：全額買入/全部賣出）
# ---------------------------

def backtest(df: pd.DataFrame, initial_capital: float = 100000) -> dict:
    df = df.copy().reset_index()
    # normalize Date column to scalar Timestamps if needed
    if 'Date' in df.columns:
        def _extract_date(x):
            if isinstance(x, pd.Series):
                # series like objects may contain the date as the first element
                val = x.iloc[0]
                return pd.to_datetime(val)
            if isinstance(x, pd.DataFrame):
                val = x.values.squeeze()[0]
                return pd.to_datetime(val)
            else:
                return pd.to_datetime(x)
        df['Date'] = df['Date'].apply(_extract_date)

    cash = initial_capital
    position = 0
    equity = []
    trades = []  # list of (entry_date, entry_price, exit_date, exit_price, qty, pnl)
    entry = None

    def to_scalar(x):
        try:
            # if pandas Series, take first element
            if isinstance(x, pd.Series):
                return float(x.iloc[0])
            return float(x)
        except Exception:
            arr = np.asarray(x)
            return arr.squeeze().item()

    for i, row in df.iterrows():
        price = to_scalar(row['Adj Close'])
        signal = row['signal']
        date = row['Date']

        # ensure signal is a scalar
        try:
            sig_arr = np.asarray(signal)
            if sig_arr.size > 1:
                sig_val = sig_arr.squeeze().item()
            else:
                sig_val = sig_arr.item()
        except Exception:
            sig_val = signal

        # buy signal and we have cash
        if sig_val == 1 and position == 0:
            qty = int(cash // price)
            if qty > 0:
                entry = {'date': date, 'price': price, 'qty': qty}
                cash -= qty * price
                position = qty
        # sell signal and we have position
        elif sig_val == -1 and position > 0:
            exit_price = to_scalar(price)
            pnl = float((exit_price - entry['price']) * position)
            trades.append({
                'entry_date': entry['date'],
                'entry_price': entry['price'],
                'exit_date': date,
                'exit_price': exit_price,
                'qty': position,
                'pnl': pnl
            })
            cash += position * exit_price
            position = 0
            entry = None

        # compute equity
        market_value = position * price
        total = cash + market_value
        equity.append({'Date': date, 'Equity': total})

    # if still holding at the end, close at last price
    if position > 0 and entry is not None:
        last_price = to_scalar(df.iloc[-1]['Adj Close'])
        last_date = df.iloc[-1]['Date']
        pnl = float((last_price - entry['price']) * position)
        trades.append({
            'entry_date': entry['date'],
            'entry_price': entry['price'],
            'exit_date': last_date,
            'exit_price': last_price,
            'qty': position,
            'pnl': pnl
        })
        cash += position * last_price
        position = 0
        equity[-1]['Equity'] = cash

    equity_df = pd.DataFrame(equity)
    equity_df['Equity'] = equity_df['Equity'].astype(float)

    metrics = compute_performance(equity_df, trades, initial_capital)
    return {'equity_df': equity_df, 'trades': trades, 'metrics': metrics}

# ---------------------------
# 績效計算
# ---------------------------

def compute_performance(equity_df: pd.DataFrame, trades: list, initial_capital: float) -> dict:
    # MDD
    equity_df['cum_max'] = equity_df['Equity'].cummax()
    equity_df['drawdown'] = (equity_df['Equity'] - equity_df['cum_max']) / equity_df['cum_max']
    mdd = equity_df['drawdown'].min() if not equity_df.empty else 0.0

    # trades metrics
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = (len(wins) / len(trades)) if trades else np.nan
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    profit_loss_ratio = (avg_win / abs(avg_loss)) if (avg_loss != 0) else np.nan

    total_return = (equity_df['Equity'].iloc[-1] / initial_capital - 1) if not equity_df.empty else 0

    return {
        'MDD': float(mdd),
        'win_rate': float(win_rate) if not np.isnan(win_rate) else None,
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_loss_ratio': float(profit_loss_ratio) if not np.isnan(profit_loss_ratio) else None,
        'total_return': float(total_return)
    }

# ---------------------------
# 圖表生成
# ---------------------------

def plot_equity(equity_df: pd.DataFrame) -> str:
    # normalize Date entries to Timestamps
    def _normalize_date(x):
        if isinstance(x, pd.Series):
            try:
                return pd.to_datetime(x.iloc[0])
            except Exception:
                return pd.to_datetime(x.values.squeeze()[0])
        if isinstance(x, pd.DataFrame):
            return pd.to_datetime(x.values.squeeze()[0])
        return pd.to_datetime(x)

    equity_df = equity_df.copy()
    equity_df['Date'] = equity_df['Date'].apply(_normalize_date)

    plt.figure(figsize=(10, 5))
    plt.plot(equity_df['Date'], equity_df['Equity'], label='Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.title('Equity Curve')
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64

# ---------------------------
# Fetch data
# ---------------------------

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # keep auto_adjust=False to preserve 'Adj Close' column if available
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError('No data fetched for {}'.format(ticker))
    # if 'Adj Close' not present, fallback to 'Close'
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            raise ValueError('Downloaded data missing Close/Adj Close for {}'.format(ticker))
    # ensure Volume exists
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df = df[['Adj Close', 'Volume']].copy()
    df.dropna(inplace=True)
    return df

# ---------------------------
# Web template (Jinja2)
# ---------------------------

HTML_TEMPLATE = Template('''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>策略回測報告</title>
  <!-- Fonts & external stylesheet -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/static/style.css">

</head>
<body>
  <div class="container">
  <h1>長短期均線 + RSI + 成交量 回測報告</h1>
  <div class="card">
    <h3>策略描述</h3>
    <p>{{ strategy_description }}</p>
  </div>
  <div class="card metrics" aria-label="key metrics">
    <div class="metric">
      <div class="label">MDD (最大回撤)</div>
      <div class="value">{{ fmt_pct(metrics.MDD,2) }}</div>
    </div>
    <div class="metric">
      <div class="label">勝率 (Win rate)</div>
      <div class="value">{{ fmt_pct(metrics.win_rate,2) if metrics.win_rate is not none else 'N/A' }}</div>
    </div>
    <div class="metric">
      <div class="label">盈虧比 (Profit/Loss Ratio)</div>
      <div class="value">{{ fmt_number(metrics.profit_loss_ratio,3) if metrics.profit_loss_ratio is not none else 'N/A' }}</div>
    </div>
    <div class="metric">
      <div class="label">總報酬 (Total Return)</div>
      <div class="value">{{ fmt_pct(metrics.total_return,2) }}</div>
    </div>
  </div>

  <div class="card">
    <h3>資金曲線 (Equity Curve)</h3>
    <img alt="Equity curve" class="equity-img" src="data:image/png;base64,{{ equity_img }}" style="max-width:100%; height:auto;" />
  <div class="card">
    <h3>回測設定</h3>
    <table class="params-table">
      <tr><th>參數</th><th>說明</th><th>數值</th></tr>
      <tr><td>Ticker</td><td>股票代號</td><td>{{ display_params.ticker }}</td></tr>
      <tr><td>Start</td><td>起始日期</td><td>{{ display_params.start }}</td></tr>
      <tr><td>End</td><td>結束日期</td><td>{{ display_params.end }}</td></tr>
      <tr><td>Capital</td><td>初始資金</td><td>{{ fmt_number(display_params.capital,0) }}</td></tr>
      <tr><td>SMA Short</td><td>短期 SMA (天)</td><td>{{ fmt_number(display_params.sma_short,0) }}</td></tr>
      <tr><td>SMA Long</td><td>長期 SMA (天)</td><td>{{ fmt_number(display_params.sma_long,0) }}</td></tr>
      <tr><td>RSI Period</td><td>RSI 週期</td><td>{{ fmt_number(display_params.rsi_period,0) }}</td></tr>
      <tr><td>RSI Buy</td><td>RSI 進場門檻</td><td>{{ fmt_number(display_params.rsi_buy,0) }}</td></tr>
      <tr><td>RSI Sell</td><td>RSI 出場門檻</td><td>{{ fmt_number(display_params.rsi_sell,0) }}</td></tr>
      <tr><td>Volume MA</td><td>成交量移動平均</td><td>{{ fmt_number(display_params.vol_ma,0) }}</td></tr>
      <tr><td>Volume Mult</td><td>量能倍率</td><td>{{ fmt_number(display_params.vol_mult,2) }}</td></tr>
    </table>
  </div>

  <div class="card">
    <h3>最近幾筆交易 (最後 10 筆)</h3>
    <table class="trades-table">
      <thead>
        <tr><th>進場日 (Entry)</th><th>進場價 (Entry Price)</th><th>出場日 (Exit)</th><th>出場價 (Exit Price)</th><th>數量 (Qty)</th><th>損益 (PnL)</th></tr>
      </thead>
      <tbody>
      {% for t in trades[-10:] %}
      <tr>
        <td>{{ t.entry_date }}</td>
        <td>{{ fmt_number(t.entry_price,3) }}</td>
        <td>{{ t.exit_date }}</td>
        <td>{{ fmt_number(t.exit_price,3) }}</td>
        <td>{{ fmt_number(t.qty,0) }}</td>
        <td class="{{ 'pnl-pos' if t.pnl > 0 else ('pnl-neg' if t.pnl < 0 else '') }}">{{ fmt_number(t.pnl,3) }}</td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <p><em>Generated by final0108.py</em></p>
  </div>
</body>
</html>
''')

# Additional index page with input form
INDEX_TEMPLATE = Template('''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>策略回測網站</title>
  <!-- Fonts & external stylesheet -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div class="container">
  <h1>策略回測網站 — 長短期均線 + RSI + 成交量</h1>
  <div class="card">
    <h3>輸入回測參數</h3>
    <form action="/run" method="get" class="grouped-form">
      <div class="form-group">
        <div class="group-header">
          <h4>基本資料</h4>
          <div class="group-meta">分類：資料欄位（選擇標的、回測期間與資金）</div>
        </div>
        <label>股票代號 (Ticker): <input name="ticker" value="{{ params.get('ticker','AAPL') }}" /></label>
        <label>起始日期 (Start YYYY-MM-DD): <input name="start" value="{{ params.get('start','2018-01-01') }}" /></label>
        <label>結束日期 (End YYYY-MM-DD): <input name="end" value="{{ params.get('end','') }}" /></label>
        <label>初始資金 (Capital): <input name="capital" value="{{ params.get('capital',100000) }}" /></label>
      </div>

      <div class="form-group">
        <div class="group-header">
          <h4>策略參數</h4>
          <div class="group-meta">分類：策略欄位（指標期數與閥值）</div>
        </div>
        <label>短期 SMA (SMA Short, 天): <input name="sma_short" value="{{ params.sma_short }}" /></label>
        <label>長期 SMA (SMA Long, 天): <input name="sma_long" value="{{ params.sma_long }}" /></label>
        <label>RSI 週期 (RSI Period): <input name="rsi_period" value="{{ params.rsi_period }}" /></label>
        <label>RSI 進場門檻 (RSI Buy): <input name="rsi_buy" value="{{ params.rsi_buy }}" /></label>
        <label>RSI 出場門檻 (RSI Sell): <input name="rsi_sell" value="{{ params.rsi_sell }}" /></label>
        <label>成交量均線 (Volume MA): <input name="vol_ma" value="{{ params.vol_ma }}" /></label>
        <label>成交量倍率 (Volume Mult): <input name="vol_mult" value="{{ params.vol_mult }}" /></label>
      </div>

      <div class="full"><button class="run-btn" type="submit">執行回測</button></div>
    </form>
  </div>
  <div class="card">
    <h3>說明</h3>
    <p>此網站可以使用自訂參數執行回測並顯示績效（MDD、勝率、盈虧比）與資金曲線。</p>
  </div>
  </div>
</body>
</html>
''')

# ---------------------------
# Run full pipeline and render HTML
# ---------------------------

def run_backtest_and_render(ticker: str, start: str, end: str, params: dict, initial_capital: float = 100000) -> str:
    df = fetch_data(ticker, start, end)
    df = generate_signals(df, params)
    bt = backtest(df, initial_capital)
    equity_img = plot_equity(bt['equity_df'])

    display_params = params.copy()
    display_params.update({'ticker': ticker, 'start': start, 'end': end, 'capital': initial_capital})

    # normalize trades for display: format dates as YYYY-MM-DD and ensure numeric types
    def _fmt_date(x):
        try:
            if isinstance(x, pd.Series):
                v = x.iloc[0]
            elif isinstance(x, pd.DataFrame):
                v = x.values.squeeze()[0]
            else:
                v = x
            ts = pd.to_datetime(v)
            return ts.strftime('%Y-%m-%d')
        except Exception:
            try:
                return str(x)
            except Exception:
                return ''

    trades_display = []
    for t in bt['trades']:
        trade = t.copy()
        trade['entry_date'] = _fmt_date(trade.get('entry_date'))
        trade['exit_date'] = _fmt_date(trade.get('exit_date'))
        # normalize numeric fields
        try:
            trade['entry_price'] = float(trade.get('entry_price', 0))
        except Exception:
            trade['entry_price'] = float(np.asarray(trade.get('entry_price', 0)).squeeze())
        try:
            trade['exit_price'] = float(trade.get('exit_price', 0))
        except Exception:
            trade['exit_price'] = float(np.asarray(trade.get('exit_price', 0)).squeeze())
        trade['qty'] = int(trade.get('qty', 0))
        try:
            trade['pnl'] = float(trade.get('pnl', 0))
        except Exception:
            trade['pnl'] = float(np.asarray(trade.get('pnl', 0)).squeeze())
        trades_display.append(trade)

    # update bt['trades'] to the display-friendly version so APIs return readable dates too
    bt['trades'] = trades_display

    # formatting helpers for templates
    def fmt_number(x, precision=2):
        try:
            if x is None:
                return 'N/A'
            if isinstance(x, int):
                return f"{x:,}"
            return f"{float(x):,.{precision}f}"
        except Exception:
            try:
                return str(x)
            except Exception:
                return 'N/A'

    def fmt_pct(x, precision=2):
        try:
            if x is None:
                return 'N/A'
            return f"{float(x) * 100:,.{precision}f}%"
        except Exception:
            return str(x)

    html = HTML_TEMPLATE.render(
        strategy_description=(
            f"使用 SMA({params['sma_short']}) 與 SMA({params['sma_long']}) 判斷趨勢，" +
            f"RSI({params['rsi_period']}) 作為動能確認，成交量超過 {params['vol_mult']} 倍 {params['vol_ma']} 日均量作為量能確認。"
        ),
        metrics=bt['metrics'],
        equity_img=equity_img,
        display_params=display_params,
        trades=bt['trades'],
        fmt_number=fmt_number,
        fmt_pct=fmt_pct
    )
    return html, bt

# ---------------------------
# FastAPI App (use lifespan handler and static files)
# ---------------------------

@asynccontextmanager
async def lifespan(app):
    # Start the background fetcher when the app starts
    asyncio.create_task(background_fetcher())
    yield

app = FastAPI(lifespan=lifespan)

# mount static files directory for CSS, fonts, etc.
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---------------------------
# Background fetching & caching
# ---------------------------
import asyncio
from typing import Dict, Any

# cache structure: ticker -> { 'last_updated': datetime, 'html': str, 'bt': dict, 'params': dict }
DATA_CACHE: Dict[str, Dict[str, Any]] = {}
TRACKED_TICKERS = set()
TICKER_LOCKS: Dict[str, asyncio.Lock] = {}
FETCH_INTERVAL = int(os.environ.get('FETCH_INTERVAL', 3600))  # seconds

async def update_ticker_cache(ticker: str, start: str = '2018-01-01', end: str = None, params: dict = None, capital: float = 100000):
    """Fetch data and run backtest for `ticker`, store results into DATA_CACHE."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if end is None:
        end = datetime.date.today().strftime('%Y-%m-%d')

    # ensure a lock exists per ticker
    lock = TICKER_LOCKS.setdefault(ticker, asyncio.Lock())
    async with lock:
        try:
            # run the blocking pipeline in a thread to avoid blocking the event loop
            html, bt = await asyncio.to_thread(run_backtest_and_render, ticker, start, end, params, capital)
            DATA_CACHE[ticker] = {
                'last_updated': datetime.datetime.now(),
                'html': html,
                'bt': bt,
                'params': {'ticker': ticker, 'start': start, 'end': end, 'capital': capital, **params}
            }
            print(f'[background] Updated cache for {ticker} at {DATA_CACHE[ticker]["last_updated"]}')
            return DATA_CACHE[ticker]
        except Exception as e:
            print(f'[background] Error updating {ticker}:', e)
            raise

async def background_fetcher():
    """Continuously fetch for tracked tickers at FETCH_INTERVAL."""
    print('[background] Background fetcher started, interval=', FETCH_INTERVAL)
    while True:
        if TRACKED_TICKERS:
            tasks = []
            for t in list(TRACKED_TICKERS):
                tasks.append(asyncio.create_task(update_ticker_cache(t)))
            # wait for all to finish (ignore exceptions to keep loop running)
            for task in tasks:
                try:
                    await task
                except Exception:
                    pass
        await asyncio.sleep(FETCH_INTERVAL)



@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    # render a simple input form using defaults
    params = DEFAULT_PARAMS.copy()
    params.update({'ticker': 'AAPL', 'start': '2018-01-01', 'end': datetime.date.today().strftime('%Y-%m-%d'), 'capital': 100000})
    return INDEX_TEMPLATE.render(params=params)


@app.get('/run', response_class=HTMLResponse)
async def run(
    ticker: str = 'AAPL',
    start: str = '2020-01-01',
    end: str = None,
    sma_short: int = 20,
    sma_long: int = 50,
    rsi_period: int = 14,
    rsi_buy: float = 55.0,
    rsi_sell: float = 45.0,
    vol_ma: int = 20,
    vol_mult: float = 1.2,
    capital: float = 100000,
):
    if end is None:
        end = datetime.date.today().strftime('%Y-%m-%d')
    params = {
        'sma_short': sma_short,
        'sma_long': sma_long,
        'rsi_period': rsi_period,
        'rsi_buy': rsi_buy,
        'rsi_sell': rsi_sell,
        'vol_ma': vol_ma,
        'vol_mult': vol_mult,
    }
    html, bt = run_backtest_and_render(ticker, start, end, params, capital)
    return HTMLResponse(content=html)

@app.get('/api/run')
async def api_run(
    ticker: str = 'AAPL',
    start: str = '2020-01-01',
    end: str = None,
    sma_short: int = 20,
    sma_long: int = 50,
    rsi_period: int = 14,
    rsi_buy: float = 55.0,
    rsi_sell: float = 45.0,
    vol_ma: int = 20,
    vol_mult: float = 1.2,
    capital: float = 100000,
):
    if end is None:
        end = datetime.date.today().strftime('%Y-%m-%d')
    params = {
        'sma_short': sma_short,
        'sma_long': sma_long,
        'rsi_period': rsi_period,
        'rsi_buy': rsi_buy,
        'rsi_sell': rsi_sell,
        'vol_ma': vol_ma,
        'vol_mult': vol_mult,
    }
    _, bt = run_backtest_and_render(ticker, start, end, params, capital)
    return {'metrics': bt['metrics'], 'trades': bt['trades']}

# ---------------------------
# Tracking endpoints
# ---------------------------

@app.post('/track')
async def track_ticker(ticker: str, start: str = '2018-01-01', end: str = None, capital: float = 100000, fetch_now: bool = True):
    """Add `ticker` to tracked list and optionally fetch immediately."""
    TRACKED_TICKERS.add(ticker.upper())
    if fetch_now:
        try:
            await update_ticker_cache(ticker.upper(), start=start, end=end, params=DEFAULT_PARAMS, capital=capital)
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    return {'status': 'ok', 'tracked': list(TRACKED_TICKERS)}

@app.post('/untrack')
async def untrack_ticker(ticker: str):
    TRACKED_TICKERS.discard(ticker.upper())
    return {'status': 'ok', 'tracked': list(TRACKED_TICKERS)}

@app.get('/tracked')
async def list_tracked():
    result = []
    for t in sorted(TRACKED_TICKERS):
        entry = DATA_CACHE.get(t)
        result.append({'ticker': t, 'last_updated': entry.get('last_updated') if entry else None})
    return {'tracked': result}

@app.get('/latest')
async def latest_report(ticker: str):
    t = ticker.upper()
    if t in DATA_CACHE:
        return HTMLResponse(content=DATA_CACHE[t]['html'])
    # else, fetch on-demand and return
    try:
        await update_ticker_cache(t)
        return HTMLResponse(content=DATA_CACHE[t]['html'])
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.post('/refresh')
async def refresh_ticker(ticker: str):
    try:
        res = await update_ticker_cache(ticker.upper())
        return {'status': 'ok', 'last_updated': res['last_updated']}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# ---------------------------
# Default params
# ---------------------------
DEFAULT_PARAMS = {
    'sma_short': 20,
    'sma_long': 50,
    'rsi_period': 14,
    'rsi_buy': 55,
    'rsi_sell': 45,
    'vol_ma': 20,
    'vol_mult': 1.2
}

# ---------------------------
# CLI 主流程
# ---------------------------

def main():
    # if --install-deps was passed at import-time, remove it so argparse doesn't fail
    if '--install-deps' in sys.argv:
        sys.argv = [a for a in sys.argv if a != '--install-deps']

    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--start', type=str, default='2018-01-01')
    parser.add_argument('--end', type=str, default=datetime.date.today().strftime('%Y-%m-%d'))
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()

    if args.serve:
        try:
            import uvicorn
        except ImportError:
            print('uvicorn is not installed. Install with: pip install uvicorn')
            return
        print('Starting server at http://127.0.0.1:8000 ...')
        uvicorn.run('final0108:app', host='127.0.0.1', port=8000, reload=True)
        return

    html, bt = run_backtest_and_render(args.ticker, args.start, args.end, DEFAULT_PARAMS, args.capital)
    # Save HTML report
    fname = f"report_{args.ticker}_{args.start}_{args.end}.html"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(html)

    print('Report saved to', fname)
    print('Metrics:')
    for k, v in bt['metrics'].items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
