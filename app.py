import os
import json
from flask import Flask, jsonify
from flask_cors import CORS
import alpaca_trade_api as tradeapi
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
try:
    alpaca_api_key = os.getenv('APCA_API_KEY_ID')
    alpaca_secret_key = os.getenv('APCA_API_SECRET_KEY')
    base_url = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url, api_version='v2')
except Exception as e:
    print(f"ERROR: Could not initialize Alpaca client. Make sure environment variables are set. Details: {e}")
    api = None

# --- SECTOR MAPPING (Simplified) ---
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
    except Exception as e:
        if fallback and os.path.exists(cache_path):
            print(f"FALLBACK: Using stale cache for '{cache_key}' due to fetch error.")
            with open(cache_path, 'r') as f:
                return json.load(f)
        # Pass the actual error message to the frontend
        return {"error": f"Failed to fetch data for {cache_key} and no cache was available. Reason: {str(e)}"}

# --- API ENDPOINTS ---
@app.route('/')
def home():
    return "Simplified Backend server is running."

def _fetch_simple_alpaca_data():
    """
    Fetches only the basic account and position info from Alpaca.
    """
    if not api:
        raise ConnectionError("Alpaca API client not initialized. Check environment variables.")
    
    account = api.get_account()
    positions = api.list_positions()

    # Basic account details
    account_data = {
        "portfolio_value": account.portfolio_value, 
        "equity": account.equity, 
        "last_equity": account.last_equity, 
        "cash": account.cash
    }

    # Basic position details
    positions_data = []
    if positions:
        asset_details = {asset.symbol: asset for asset in api.list_assets(status='active')}
        for p in positions:
            asset = asset_details.get(p.symbol)
            industry = asset.industry if asset and hasattr(asset, 'industry') and asset.industry else 'Other'
            sector = SECTOR_MAPPING.get(industry, industry)
            
            positions_data.append({
                "symbol": p.symbol, 
                "qty": p.qty, 
                "market_value": p.market_value, 
                "unrealized_intraday_pl": p.unrealized_intraday_pl, 
                "unrealized_pl": p.unrealized_pl, 
                "unrealized_plpc": p.unrealized_plpc, 
                "side": p.side,
                "sector": sector
            })

    return {"account": account_data, "positions": positions_data}

@app.route('/api/alpaca/portfolio')
def get_alpaca_portfolio():
    portfolio_data = _get_cached_or_fetch("simple_alpaca_portfolio", _fetch_simple_alpaca_data)
    if "error" in portfolio_data:
        return jsonify(portfolio_data), 500
    return jsonify(portfolio_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

