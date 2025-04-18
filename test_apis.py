# test_apis.py
import requests
import json
from datetime import datetime, timedelta

# --- Configuration ---
# URLs and Credentials (Match your main scripts)
API_BASE_URL_STOCK = "http://140.116.86.242:8081" # :8081
API_BASE_URL_HISTORY = "http://140.116.86.241:8800" # :8800
REQUEST_TIMEOUT = 25
# --- Account Info ---
USER_ACCOUNT = "N26132089"     # <<< REPLACE with your account for :8081 API >>>
USER_PASSWORD = "joshua900905"    # <<< REPLACE with your password for :8081 API >>>
TEAM_ACCOUNT_ID = "team1"        # <<< REPLACE with your account/team for :8800 API >>>
# --- Test Parameters ---
TEST_STOCK_CODE = "2330" # Stock code to test history/bargain data
TEST_HISTORY_DAYS = 5     # How many days back to test Get_Stock_Informations

# === Generic API Request Function ===
def api_request(base_url, method, endpoint, data=None, description="API Request"):
    """Generic function to handle API requests."""
    url = f"{base_url}/{endpoint}"
    print(f"\n--- Testing {description} ---")
    print(f"URL: {url}")
    if data: print(f"Data: {data}")
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        elif method.upper() == 'POST':
            response = requests.post(url, data=data, timeout=REQUEST_TIMEOUT)
        else:
            print(f"ERROR: Unsupported method {method}")
            return None

        print(f"Status Code: {response.status_code}")
        # Try to print raw text for debugging, limit length
        print(f"Raw Response (first 500 chars): {response.text[:500]}")

        response.raise_for_status() # Raise HTTPError for bad responses

        # Try parsing JSON
        try:
            json_response = response.json()
            print("Response JSON (parsed successfully):")
            # Pretty print JSON
            print(json.dumps(json_response, indent=2, ensure_ascii=False))
            return json_response
        except json.JSONDecodeError as json_err:
            print(f"ERROR: Failed to decode JSON response: {json_err}")
            return {"error": "JSON Decode Error", "raw_response": response.text} # Return raw text on decode error

    except requests.exceptions.Timeout:
        print(f"ERROR: API request timed out ({REQUEST_TIMEOUT}s)")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"ERROR: API HTTP Error: {http_err}")
        return None # Usually indicates client/server error
    except requests.exceptions.RequestException as req_err:
        print(f"ERROR: API Request Failed: {req_err}")
        return None
    except Exception as e:
        print(f"ERROR: Unknown error during API request: {e}")
        return None
    finally:
        print("-" * (len(description) + 10))


# === Test Functions ===

def test_get_stock_informations():
    """Tests fetching OHLCV data from :8081 API."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TEST_HISTORY_DAYS)
    end_date_str = end_date.strftime('%Y%m%d')
    start_date_str = start_date.strftime('%Y%m%d')
    endpoint = f'stock/api/v1/api_get_stock_info_from_date_json/{TEST_STOCK_CODE}/{start_date_str}/{end_date_str}'
    result = api_request(API_BASE_URL_STOCK, 'GET', endpoint, description=f"Get Stock Info ({TEST_STOCK_CODE})")
    if result and isinstance(result.get('data'), list):
        print(f"[OK] Get Stock Info seems successful. Found {len(result['data'])} records.")
        if result['data']:
             print("First record sample:")
             print(json.dumps(result['data'][0], indent=2, ensure_ascii=False))
    else:
        print("[FAIL] Get Stock Info failed or returned unexpected format.")

def test_get_daily_bargain_data():
    """Tests fetching daily bargain data (incl. 'change') from :8800 API."""
    query_date = datetime.now() - timedelta(days=1) # Test for yesterday
    # TODO: Need logic here to find previous *trading* day if yesterday was holiday/weekend
    query_date_str = query_date.strftime('%Y%m%d')
    endpoint = f'api/v1/bargain_data/{TEST_STOCK_CODE}/{query_date_str}/{query_date_str}'
    result = api_request(API_BASE_URL_HISTORY, 'GET', endpoint, description=f"Get Daily Bargain Data ({TEST_STOCK_CODE} for {query_date_str})")
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        print(f"[OK] Get Daily Bargain Data seems successful for {query_date_str}.")
        print("First record sample:")
        print(json.dumps(result[0], indent=2, ensure_ascii=False))
        # Check specifically for OHLC keys needed later
        required_keys = ['Open', 'High', 'Low', 'Close', 'change']
        if all(key in result[0] for key in required_keys):
             print("[OK] Bargain data contains required OHLC and change keys.")
        else:
             print(f"[WARN] Bargain data might be missing required keys: {required_keys}")
    elif isinstance(result, list) and len(result) == 0:
         print(f"[INFO] Get Daily Bargain Data returned empty list for {query_date_str} (Maybe non-trading day?).")
    else:
        print("[FAIL] Get Daily Bargain Data failed or returned unexpected format.")

def test_get_transaction_history():
    """Tests fetching transaction history from :8800 API."""
    query_date = datetime.now() - timedelta(days=1) # Test for yesterday
    # TODO: Adjust for non-trading days if needed
    query_date_str = query_date.strftime('%Y%m%d')
    endpoint = f'api/v1/transaction_history/{TEAM_ACCOUNT_ID}/{query_date_str}/{query_date_str}'
    result = api_request(API_BASE_URL_HISTORY, 'GET', endpoint, description=f"Get Transaction History ({TEAM_ACCOUNT_ID} for {query_date_str})")
    if isinstance(result, list):
        print(f"[OK] Get Transaction History seems successful. Found {len(result)} raw records for {query_date_str}.")
        if result:
             print("First record sample:")
             print(json.dumps(result[0], indent=2, ensure_ascii=False))
             # Check for essential keys
             if all(key in result[0] for key in ['stock_code', 'type', 'price', 'state']):
                 print("[OK] Transaction record contains essential keys.")
             else:
                 print("[WARN] Transaction record might be missing essential keys.")
    else:
        print("[FAIL] Get Transaction History failed or returned unexpected format.")

def test_get_user_stocks():
    """Tests fetching user holdings from :8081 API."""
    endpoint = 'stock/api/v1/get_user_stocks'
    data = {'account': USER_ACCOUNT, 'password': USER_PASSWORD}
    result = api_request(API_BASE_URL_STOCK, 'POST', endpoint, data=data, description="Get User Stocks")
    if result and result.get('result') == 'success' and isinstance(result.get('data'), list):
        print(f"[OK] Get User Stocks successful. Found {len(result['data'])} holding(s).")
        if result['data']:
             print("First holding sample:")
             print(json.dumps(result['data'][0], indent=2, ensure_ascii=False))
             # Check for essential keys
             if 'stock_code_id' in result['data'][0] or 'stock_code' in result['data'][0]:
                 print("[OK] Holding record contains stock code identifier.")
             else:
                 print("[WARN] Holding record might be missing stock code identifier.")
             if 'shares' in result['data'][0] or 'stock_shares' in result['data'][0]:
                 print("[OK] Holding record contains shares information.")
             else:
                  print("[WARN] Holding record might be missing shares information.")

    elif result and result.get('result') == 'failed':
         print(f"[FAIL] Get User Stocks failed (API Error): {result.get('status', 'Unknown')}")
    else:
        print("[FAIL] Get User Stocks failed or returned unexpected format.")


# === Run Tests ===
if __name__ == "__main__":
    print("Starting API Tests...")
    print("="*40)

    test_get_stock_informations() # Tests :8081 history endpoint
    print("="*40)
    time.sleep(1) # Small delay

    test_get_daily_bargain_data() # Tests :8800 bargain endpoint
    print("="*40)
    time.sleep(1)

    test_get_transaction_history() # Tests :8800 history endpoint
    print("="*40)
    time.sleep(1)

    test_get_user_stocks() # Tests :8081 holdings endpoint
    print("="*40)

    print("\nAPI Tests Finished.")