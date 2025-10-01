import os
import json
from flask import Flask, jsonify
from flask_cors import CORS
import alpaca_trade_api as tradeapi
from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
CACHE_DIR = "cache"
CACHE_DURATION_SECONDS = 86400  # 24 hours

# Create cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- API CLIENTS SETUP ---
# It's crucial that these environment variables are set in your deployment environment (e.g., Render).
try:
    alpaca_api_key = os.getenv('APCA_API_KEY_ID')
    alpaca_secret_key = os.getenv('APCA_API_SECRET_KEY')
    fred_api_key = os.getenv('FRED_API_KEY')

    # Use paper trading endpoint for development/testing
    base_url = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url, api_version='v2')
    fred = Fred(api_key=fred_api_key)
except Exception as e:
    print(f"ERROR: Could not initialize API clients. Make sure environment variables are set. Details: {e}")
    api = None
    fred = None

# --- CACHING LOGIC ---
def _fetch_and_cache(cache_key, fetch_function):
    """
    Executes the fetch_function, saves its result to a cache file, and returns the result.
    """
    print(f"CACHE MISS: Fetching fresh data for '{cache_key}'...")
    try:
        data = fetch_function()
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"API FETCH ERROR for '{cache_key}': {e}")
        return None

def _get_cached_or_fetch(cache_key, fetch_function, fallback=True):
    """
    Tries to retrieve data from cache. If it's stale or doesn't exist, it calls the fetch_function.
    If the fetch fails, it can fall back to using stale cache data.
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if os.path.exists(cache_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_mod_time < timedelta(seconds=CACHE_DURATION_SECONDS):
            print(f"CACHE HIT: Using recent data for '{cache_key}'.")
            with open(cache_path, 'r') as f:
                return json.load(f)

    # If cache is stale or doesn't exist, fetch new data
    fresh_data = _fetch_and_cache(cache_key, fetch_function)

    if fresh_data is not None:
        return fresh_data
    elif fallback and os.path.exists(cache_path):
        # If fetch failed, fall back to stale cache if it exists
        print(f"FALLBACK: Using stale cache for '{cache_key}' due to fetch error.")
        with open(cache_path, 'r') as f:
            return json.load(f)
    else:
        # If fetch failed and there's no cache to fall back on
        return {"error": f"Failed to fetch data for {cache_key} and no cache was available."}

# --- API ENDPOINTS ---

@app.route('/')
def home():
    return "Backend server is running."

def _fetch_alpaca_portfolio():
    """
    Internal function to fetch all necessary Alpaca data: account, positions, and historical bars.
    """
    if not api:
        raise ConnectionError("Alpaca API client is not initialized.")

    account = api.get_account()
    positions = api.list_positions()

    # Prepare data to be JSON serializable
    account_data = {
        "portfolio_value": account.portfolio_value,
        "equity": account.equity,
        "last_equity": account.last_equity,
        "cash": account.cash
    }
    
    positions_data = [
        {
            "symbol": p.symbol,
            "qty": p.qty,
            "market_value": p.market_value,
            "unrealized_intraday_pl": p.unrealized_intraday_pl,
            "unrealized_pl": p.unrealized_pl,
            "unrealized_plpc": p.unrealized_plpc,
            "side": p.side,
            "asset_id": p.asset_id
        } for p in positions
    ]

    # Fetch asset details and historical bars only if there are positions
    bars_data = {}
    if positions:
        symbols = [p.symbol for p in positions]
        
        # Add sector information
        asset_details = {asset.symbol: asset for asset in api.list_assets(status='active')}
        for pos in positions_data:
            pos['sector'] = asset_details.get(pos['symbol'], {}).sector or 'Other'
            
        # Fetch historical data
        end_date = pd.Timestamp.now(tz='America/New_York').isoformat()
        start_date = (pd.Timestamp.now(tz='America/New_York') - pd.Timedelta(days=365*2)).isoformat() # Approx 2 years for 252 trading days
        
        barset = api.get_bars(symbols, "1Day", start=start_date, end=end_date, adjustment='split').df
        # Reformat the pandas DataFrame into the JSON structure the frontend expects
        for symbol in symbols:
            if symbol in barset.index.get_level_values('symbol'):
                symbol_bars = barset.loc[symbol]
                bars_data[symbol] = [
                    {"t": bar.name.strftime('%Y-%m-%dT%H:%M:%SZ'), "c": bar.close} for bar in symbol_bars.itertuples()
                ]

    return {"account": account_data, "positions": positions_data, "bars": bars_data}


@app.route('/api/alpaca/portfolio')
def get_alpaca_portfolio():
    """
    Provides all portfolio data (account, positions, bars) in a single call, with caching.
    """
    portfolio_data = _get_cached_or_fetch("alpaca_portfolio", _fetch_alpaca_portfolio)
    if "error" in portfolio_data:
        return jsonify(portfolio_data), 500
    return jsonify(portfolio_data)


def _fetch_fred_series(series_id):
    """
    Internal function to fetch a specific FRED series.
    """
    if not fred:
        raise ConnectionError("FRED API client is not initialized.")
    data = fred.get_series(series_id)
    # Convert pandas Series to a list of dicts {x: date, y: value}
    return [
        {"x": index.strftime('%Y-%m-%d'), "y": value}
        for index, value in data.items() if pd.notna(value)
    ]

@app.route('/api/fred/series/<series_id>')
def get_fred_series(series_id):
    """
    Provides data for a given FRED series ID, with caching.
    """
    # Use lambda to pass series_id to the fetch function
    series_data = _get_cached_or_fetch(f"fred_{series_id}", lambda: _fetch_fred_series(series_id))
    if "error" in series_data:
        return jsonify(series_data), 500
    return jsonify(series_data)


if __name__ == "__main__":
    # Use a port that Render will provide via the PORT environment variable,
    # default to 5000 for local development.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

