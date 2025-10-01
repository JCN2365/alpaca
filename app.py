import os
import json
from flask import Flask, jsonify
from flask_cors import CORS
import alpaca_trade_api as tradeapi
from fredapi import Fred
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
CACHE_DIR = "cache"
CACHE_DURATION_SECONDS = 86400  # 24 hours
BENCHMARK_SYMBOL = 'SPY'

# Create cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- API CLIENTS SETUP ---
try:
    alpaca_api_key = os.getenv('APCA_API_KEY_ID')
    alpaca_secret_key = os.getenv('APCA_API_SECRET_KEY')
    fred_api_key = os.getenv('FRED_API_KEY')

    base_url = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url, api_version='v2')
    fred = Fred(api_key=fred_api_key)
except Exception as e:
    print(f"ERROR: Could not initialize API clients. Make sure environment variables are set. Details: {e}")
    api = None
    fred = None

# --- SECTOR MAPPING ---
SECTOR_MAPPING = {
    'Software': 'Technology', 'Semiconductors': 'Technology', 'Technology Hardware': 'Technology',
    'IT Services': 'Technology', 'Cloud Computing': 'Technology', 'Banks': 'Financial Services',
    'Capital Markets': 'Financial Services', 'Insurance': 'Financial Services',
    'Financial Technology (FinTech)': 'Financial Services', 'Pharmaceuticals': 'Healthcare',
    'Biotechnology': 'Healthcare', 'Medical Devices': 'Healthcare', 'Healthcare Providers': 'Healthcare',
    'Automobiles': 'Consumer Cyclical', 'Retail': 'Consumer Cyclical', 'Media': 'Consumer Cyclical',
    'Hotels, Restaurants & Leisure': 'Consumer Cyclical', 'Food & Staples Retailing': 'Consumer Defensive',
    'Beverages': 'Consumer Defensive', 'Household Products': 'Consumer Defensive', 'Aerospace & Defense': 'Industrials',
    'Machinery': 'Industrials', 'Airlines': 'Industrials', 'Transportation': 'Industrials',
    'Oil & Gas': 'Energy', 'Renewable Energy': 'Energy', 'Chemicals': 'Basic Materials',
    'Metals & Mining': 'Basic Materials', 'REITs': 'Real Estate', 'Utilities': 'Utilities',
    'Telecommunication Services': 'Communication Services', 'Entertainment': 'Communication Services',
}


# --- CACHING LOGIC ---
def _fetch_and_cache(cache_key, fetch_function):
    print(f"CACHE MISS: Fetching fresh data for '{cache_key}'...")
    try:
        data = fetch_function()
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"API FETCH ERROR for '{cache_key}': {e}")
        raise e

def _get_cached_or_fetch(cache_key, fetch_function, fallback=True):
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_mod_time < timedelta(seconds=CACHE_DURATION_SECONDS):
            print(f"CACHE HIT: Using recent data for '{cache_key}'.")
            with open(cache_path, 'r') as f:
                return json.load(f)
    try:
        fresh_data = _fetch_and_cache(cache_key, fetch_function)
        return fresh_data
    except Exception:
        if fallback and os.path.exists(cache_path):
            print(f"FALLBACK: Using stale cache for '{cache_key}' due to fetch error.")
            with open(cache_path, 'r') as f:
                return json.load(f)
        return {"error": f"Failed to fetch data for {cache_key} and no cache was available."}

# --- ADVANCED METRICS CALCULATION ---
def _calculate_advanced_metrics(positions_data, bars_data):
    """
    Calculates advanced performance and risk metrics for the portfolio.
    """
    if not positions_data or not bars_data:
        return {}

    # Create a DataFrame of historical prices for all positions
    portfolio_prices = pd.DataFrame({
        symbol: {pd.to_datetime(bar['t']): bar['c'] for bar in bars_data.get(symbol, [])}
        for symbol in [p['symbol'] for p in positions_data]
    }).ffill()

    # Drop columns that are all NaN (assets with no historical data)
    portfolio_prices.dropna(axis=1, how='all', inplace=True)

    if portfolio_prices.empty:
        return {}

    # Calculate daily returns
    returns = portfolio_prices.pct_change().dropna()

    # Get current market values and calculate weights
    market_values = pd.Series({p['symbol']: float(p['market_value']) for p in positions_data if p['symbol'] in returns.columns})
    total_market_value = market_values.sum()
    if total_market_value == 0: return {}
    
    weights = market_values / total_market_value

    # Calculate portfolio daily returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # --- Fetch Benchmark Data (SPY) ---
    end_date = pd.Timestamp.now(tz='America/New_York').isoformat()
    start_date = (pd.Timestamp.now(tz='America/New_York') - pd.Timedelta(days=365*2)).isoformat()
    try:
        spy_bars = api.get_bars(BENCHMARK_SYMBOL, "1Day", start=start_date, end=end_date, adjustment='split').df
        spy_returns = spy_bars['close'].pct_change().dropna()
    except Exception as e:
        print(f"Could not fetch benchmark data for {BENCHMARK_SYMBOL}: {e}")
        spy_returns = pd.Series() # Return empty series on failure

    # Align portfolio and benchmark returns
    aligned_returns, aligned_spy_returns = portfolio_returns.align(spy_returns, join='inner')

    if aligned_returns.empty or len(aligned_returns) < 2:
        return {}

    # --- Risk-Adjusted Performance ---
    risk_free_rate = 0.05 / 252  # Approximate daily risk-free rate
    
    # Sharpe Ratio
    sharpe_ratio = (aligned_returns.mean() - risk_free_rate) / aligned_returns.std() * np.sqrt(252) if aligned_returns.std() != 0 else 0

    # Sortino Ratio
    downside_returns = aligned_returns[aligned_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (aligned_returns.mean() - risk_free_rate) / downside_std * np.sqrt(252) if downside_std != 0 else 0

    # --- Alpha and Beta ---
    alpha = 0
    beta = 0
    if not aligned_spy_returns.empty and len(aligned_spy_returns) > 1:
        covariance = aligned_returns.cov(aligned_spy_returns)
        benchmark_variance = aligned_spy_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        alpha_daily = (aligned_returns.mean() - risk_free_rate) - beta * (aligned_spy_returns.mean() - risk_free_rate)
        alpha = alpha_daily * 252

    # --- Volatility and Drawdown ---
    volatility_annualized = aligned_returns.std() * np.sqrt(252)
    
    # Max Drawdown
    cumulative_returns = (1 + aligned_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    current_drawdown = drawdown.iloc[-1] if not drawdown.empty else 0
    
    # Value at Risk (VaR) - 95% confidence (historical method)
    var_95 = aligned_returns.quantile(0.05)
    
    return {
        "sharpe_ratio": round(sharpe_ratio, 3),
        "sortino_ratio": round(sortino_ratio, 3),
        "alpha_annualized": round(alpha, 4),
        "beta": round(beta, 3),
        "volatility_annualized": round(volatility_annualized, 4),
        "max_drawdown": round(max_drawdown, 4),
        "current_drawdown": round(current_drawdown, 4),
        "var_95_historical": round(var_95, 4)
    }

# --- API ENDPOINTS ---
@app.route('/')
def home():
    return "Backend server is running."

def _fetch_alpaca_portfolio():
    if not api: raise ConnectionError("Alpaca API client not initialized.")
    account = api.get_account()
    positions = api.list_positions()

    account_data = {"portfolio_value": account.portfolio_value, "equity": account.equity, "last_equity": account.last_equity, "cash": account.cash}
    positions_data = [{"symbol": p.symbol, "qty": p.qty, "market_value": p.market_value, "unrealized_intraday_pl": p.unrealized_intraday_pl, "unrealized_pl": p.unrealized_pl, "unrealized_plpc": p.unrealized_plpc, "side": p.side, "asset_id": p.asset_id} for p in positions]

    bars_data = {}
    if positions:
        symbols = [p.symbol for p in positions]
        asset_details = {asset.symbol: asset for asset in api.list_assets(status='active')}
        for pos in positions_data:
            asset = asset_details.get(pos['symbol'])
            pos['sector'] = SECTOR_MAPPING.get(asset.industry, asset.industry) if asset and hasattr(asset, 'industry') and asset.industry else 'Other'
        
        end_date = pd.Timestamp.now(tz='America/New_York').isoformat()
        start_date = (pd.Timestamp.now(tz='America/New_York') - pd.Timedelta(days=365*2)).isoformat()
        barset = api.get_bars(symbols, "1Day", start=start_date, end=end_date, adjustment='split').df
        for symbol in symbols:
            if not barset.empty and symbol in barset.index.get_level_values('symbol'):
                symbol_bars = barset.loc[symbol]
                bars_data[symbol] = [{"t": bar.name.strftime('%Y-%m-%dT%H:%M:%SZ'), "c": bar.close} for bar in symbol_bars.itertuples()]

    # Calculate and include advanced metrics
    advanced_metrics = _calculate_advanced_metrics(positions_data, bars_data)

    return {"account": account_data, "positions": positions_data, "bars": bars_data, "metrics": advanced_metrics}

@app.route('/api/alpaca/portfolio')
def get_alpaca_portfolio():
    portfolio_data = _get_cached_or_fetch("alpaca_portfolio", _fetch_alpaca_portfolio)
    if "error" in portfolio_data: return jsonify(portfolio_data), 500
    return jsonify(portfolio_data)

def _fetch_fred_series(series_id):
    if not fred: raise ConnectionError("FRED API client is not initialized.")
    data = fred.get_series(series_id)
    return [{"x": index.strftime('%Y-%m-%d'), "y": value} for index, value in data.items() if pd.notna(value)]

@app.route('/api/fred/series/<series_id>')
def get_fred_series(series_id):
    series_data = _get_cached_or_fetch(f"fred_{series_id}", lambda: _fetch_fred_series(series_id))
    if "error" in series_data: return jsonify(series_data), 500
    return jsonify(series_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
