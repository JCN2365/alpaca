import os
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify
from flask_cors import CORS
import alpaca_trade_api as tradeapi
from fredapi import Fred
import pandas as pd

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- Caching Configuration ---
CACHE_DIR = "cache"
CACHE_EXPIRY_SECONDS = 86400  # 24 hours

# Ensure the cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- API Configuration ---
# Securely load API keys from environment variables set on Render
alpaca_api_key = os.getenv('APCA_API_KEY_ID')
alpaca_api_secret = os.getenv('APCA_API_SECRET_KEY')
alpaca_base_url = 'https://paper-api.alpaca.markets'
fred_api_key = os.getenv('FRED_API_KEY')

# --- API Client Initialization ---
alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_api_secret, alpaca_base_url, api_version='v2') if alpaca_api_key and alpaca_api_secret else None
fred = Fred(api_key=fred_api_key) if fred_api_key else None


# --- Caching Helper Functions ---

def get_cached_data(cache_filename):
    """
    Reads data from a cache file if it exists and is not expired.
    Returns the cached data or None.
    """
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as f:
            cache = json.load(f)
            timestamp = datetime.fromisoformat(cache['timestamp'])
            if datetime.utcnow() - timestamp < timedelta(seconds=CACHE_EXPIRY_SECONDS):
                return cache['data'] # Cache is valid
    return None

def write_cached_data(cache_filename, data):
    """
    Writes new data and a current timestamp to a cache file.
    """
    cache_content = {
        'timestamp': datetime.utcnow().isoformat(),
        'data': data
    }
    with open(cache_filename, 'w') as f:
        json.dump(cache_content, f)
        
def get_fallback_data(cache_filename):
    """
    Reads stale data from a cache file if it exists, used when API calls fail.
    """
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as f:
            return json.load(f)['data']
    return None

# --- API Endpoints ---

@app.route('/')
def home():
    return "Backend server with daily caching is running."

@app.route('/api/alpaca/positions', methods=['GET'])
def get_alpaca_positions():
    """
    Fetches Alpaca positions, using a daily cache with fallback.
    """
    cache_file = os.path.join(CACHE_DIR, "alpaca_positions.json")
    
    # 1. Try to get fresh, valid cache
    cached_data = get_cached_data(cache_file)
    if cached_data:
        return jsonify(cached_data)

    # 2. If cache is expired or missing, fetch from API
    if not alpaca_api:
        return jsonify({"error": "Alpaca API keys not configured"}), 500
        
    try:
        positions = alpaca_api.list_positions()
        positions_data = [p._asdict() for p in positions]
        write_cached_data(cache_file, positions_data) # Update cache
        return jsonify(positions_data)
    except Exception as e:
        # 3. If API fails, try to serve stale data as a fallback
        fallback_data = get_fallback_data(cache_file)
        if fallback_data:
            return jsonify(fallback_data)
        else:
            # Only fail if API is down AND there's no cache at all
            return jsonify({"error": f"API fetch failed and no cache available: {str(e)}"}), 500

@app.route('/api/fred/series/<series_id>', methods=['GET'])
def get_fred_series(series_id):
    """
    Fetches FRED series data, using a daily cache with fallback.
    """
    cache_file = os.path.join(CACHE_DIR, f"fred_{series_id}.json")

    # 1. Try to get fresh, valid cache
    cached_data = get_cached_data(cache_file)
    if cached_data:
        return jsonify(cached_data)

    # 2. If cache is expired or missing, fetch from API
    if not fred:
        return jsonify({"error": "FRED API key not configured"}), 500

    try:
        data = fred.get_series(series_id)
        data_cleaned = data.dropna()
        formatted_data = [{"x": index.strftime('%Y-%m-%d'), "y": value} for index, value in data_cleaned.items()]
        write_cached_data(cache_file, formatted_data) # Update cache
        return jsonify(formatted_data)
    except Exception as e:
        # 3. If API fails, try to serve stale data as a fallback
        fallback_data = get_fallback_data(cache_file)
        if fallback_data:
            return jsonify(fallback_data)
        else:
            # Only fail if API is down AND there's no cache at all
            return jsonify({"error": f"API fetch failed for '{series_id}' and no cache available: {str(e)}"}), 404

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
