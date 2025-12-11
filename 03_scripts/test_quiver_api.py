#!/usr/bin/env python3
"""Diagnostic script to test Quiver API integration and troubleshoot empty trades."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Also add core to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    print("⚠️  WARNING: requests library not installed")


def test_token_configuration():
    """Test 1: Check if Quiver API token is configured."""
    print("\n" + "="*70)
    print("TEST 1: Quiver API Token Configuration")
    print("="*70)

    token_sources = []

    # Check environment variable
    env_token = os.getenv("QUIVER_API_TOKEN") or os.getenv("QQ_API_TOKEN")
    if env_token:
        token_sources.append(("Environment Variable", env_token[:10] + "..." + env_token[-4:] if len(env_token) > 14 else env_token))

    # Check token file in data dir
    data_token_path = Path(__file__).parent.parent / "01_data" / "quiver_token.txt"
    if data_token_path.exists():
        try:
            token = data_token_path.read_text(encoding="utf-8").strip()
            if token:
                token_sources.append(("Data Dir File", data_token_path, token[:10] + "..." + token[-4:] if len(token) > 14 else token))
        except Exception as e:
            print(f"  ⚠️  Could not read {data_token_path}: {e}")

    # Check token file in core
    core_token_path = Path(__file__).parent / "quiver_token.txt"
    if core_token_path.exists():
        try:
            token = core_token_path.read_text(encoding="utf-8").strip()
            if token:
                token_sources.append(("Core Dir File", core_token_path, token[:10] + "..." + token[-4:] if len(token) > 14 else token))
        except Exception as e:
            print(f"  ⚠️  Could not read {core_token_path}: {e}")

    # Check current directory
    cwd_token_path = Path.cwd() / "quiver_token.txt"
    if cwd_token_path.exists():
        try:
            token = cwd_token_path.read_text(encoding="utf-8").strip()
            if token:
                token_sources.append(("Current Dir File", cwd_token_path, token[:10] + "..." + token[-4:] if len(token) > 14 else token))
        except Exception as e:
            print(f"  ⚠️  Could not read {cwd_token_path}: {e}")

    if not token_sources:
        print("  ❌ NO TOKEN FOUND")
        print("\n  Token should be in one of these locations:")
        print(f"    - Environment variable: QUIVER_API_TOKEN")
        print(f"    - File: {data_token_path}")
        print(f"    - File: {core_token_path}")
        print(f"    - File: {cwd_token_path}")
        print("\n  To fix:")
        print("    echo 'your-token-here' > 01_data/quiver_token.txt")
        return None

    print(f"  ✅ Found {len(token_sources)} token source(s):\n")
    for i, source in enumerate(token_sources, 1):
        if len(source) == 2:
            print(f"    {i}. {source[0]}: {source[1]}")
        else:
            print(f"    {i}. {source[0]}: {source[1]}")
            print(f"       Value: {source[2]}")

    # Return the first token for testing
    if len(token_sources[0]) == 2:
        return token_sources[0][1]
    else:
        token_path = token_sources[0][1]
        return token_path.read_text(encoding="utf-8").strip()


def test_api_direct(token: str):
    """Test 2: Make direct API call to Quiver."""
    print("\n" + "="*70)
    print("TEST 2: Direct Quiver API Call")
    print("="*70)

    if not REQUESTS_OK:
        print("  ❌ requests library not available")
        return None

    # Test with Cory Booker (known active trader)
    test_bioguide = "B001288"
    test_name = "Cory Booker"

    print(f"\n  Testing with: {test_name} (bioguide_id={test_bioguide})")
    print(f"  API: https://api.quiverquant.com/beta/bulk/congresstrading")
    print(f"  Token: {token[:10]}...{token[-4:]}\n")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    params = {"bioguide_id": test_bioguide}

    try:
        print("  Making API request...")
        response = requests.get(
            "https://api.quiverquant.com/beta/bulk/congresstrading",
            headers=headers,
            params=params,
            timeout=10.0
        )

        print(f"  Response Status: {response.status_code}")
        print(f"  Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            print("  ✅ API call successful!\n")

            try:
                data = response.json()
                print(f"  Response Type: {type(data)}")
                print(f"  Number of trades: {len(data) if isinstance(data, list) else 'N/A'}")

                if isinstance(data, list):
                    if len(data) > 0:
                        print(f"\n  ✅ Found {len(data)} trades for {test_name}!\n")
                        print("  First 3 trades:")
                        for i, trade in enumerate(data[:3], 1):
                            ticker = trade.get("Ticker", "N/A")
                            txn = trade.get("Transaction", "N/A")
                            traded = trade.get("Traded", "N/A")
                            filed = trade.get("Filed", "N/A")
                            amount = trade.get("Range") or trade.get("Amount", "N/A")
                            print(f"    {i}. {traded}: {ticker} - {txn} ({amount})")
                            print(f"       Filed: {filed}")
                        return data
                    else:
                        print(f"\n  ⚠️  API returned EMPTY list for {test_name}")
                        print("  This means the API is working but this member has no recent trades.")
                        print("  This is NORMAL and expected for many congress members.\n")
                        return []
                else:
                    print(f"\n  ⚠️  Unexpected response type: {type(data)}")
                    print(f"  Response: {json.dumps(data, indent=2)[:500]}")
                    return None

            except ValueError as e:
                print(f"  ❌ Response is not valid JSON: {e}")
                print(f"  Response text: {response.text[:500]}")
                return None

        elif response.status_code == 401:
            print("  ❌ Authentication failed (401 Unauthorized)")
            print("  Your token may be invalid or expired.")
            print("  Please check your Quiver account and regenerate token if needed.")
            return None

        elif response.status_code == 403:
            print("  ❌ Access forbidden (403 Forbidden)")
            print("  Your token may not have permission to access this endpoint.")
            print("  Check your Quiver subscription plan.")
            return None

        elif response.status_code == 429:
            print("  ❌ Rate limit exceeded (429 Too Many Requests)")
            print("  You've made too many API calls. Wait before trying again.")
            return None

        else:
            print(f"  ❌ API error: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return None

    except requests.Timeout:
        print("  ❌ API request timed out after 10 seconds")
        return None
    except requests.ConnectionError as e:
        print(f"  ❌ Connection error: {e}")
        print("  Check your internet connection.")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_module_integration():
    """Test 3: Test integration with overlay.py module."""
    print("\n" + "="*70)
    print("TEST 3: Module Integration Test")
    print("="*70)

    try:
        from quiver_api import fetch_congress_trading, load_token
        print("  ✅ quiver_api module imported successfully")

        # Try to load token via module
        try:
            token = load_token()
            print(f"  ✅ Token loaded via module: {token[:10]}...{token[-4:]}")
        except Exception as e:
            print(f"  ⚠️  Could not load token via module: {e}")
            return False

        # Try to fetch trades
        test_bioguide = "B001288"  # Cory Booker
        print(f"\n  Testing fetch_congress_trading('{test_bioguide}')...")

        try:
            trades = fetch_congress_trading(test_bioguide, token=token)
            print(f"  ✅ API call successful via module")
            print(f"  Returned {len(trades)} trades")

            if trades:
                print("\n  Sample trade:")
                print(f"    {json.dumps(trades[0], indent=4)}")

            return True

        except Exception as e:
            print(f"  ❌ API call failed: {e}")
            return False

    except ImportError as e:
        print(f"  ❌ Could not import quiver_api module: {e}")
        return False


def test_cache_status():
    """Test 4: Check cache file status."""
    print("\n" + "="*70)
    print("TEST 4: Cache File Status")
    print("="*70)

    cache_path = Path(__file__).parent.parent / "02_outputs" / "quiver_trades_cache.json"

    print(f"\n  Cache file: {cache_path}")

    if not cache_path.exists():
        print("  ⚠️  Cache file does not exist (will be created on first API call)")
        return None

    print("  ✅ Cache file exists")

    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)

        print(f"  Cache entries: {len(cache)}")
        print(f"  File size: {cache_path.stat().st_size:,} bytes")
        print(f"  Last modified: {time.ctime(cache_path.stat().st_mtime)}")

        if not cache:
            print("\n  Cache is empty (no trades fetched yet)")
            return cache

        print("\n  Cached members:")
        now = time.time()

        for i, (key, entry) in enumerate(cache.items(), 1):
            age = now - entry.get('ts', 0)
            ttl = entry.get('ttl', 1800)
            expired = age > ttl
            trades_count = len(entry.get('data', {}).get('trades', []))

            # Parse key
            parts = key.split('|')
            if len(parts) >= 2:
                bioguide = parts[1]
            else:
                bioguide = "unknown"

            status = "EXPIRED" if expired else "VALID"
            print(f"    {i}. {bioguide}: {trades_count} trades, {age:.0f}s old [{status}]")

        # Count empty vs non-empty
        empty_count = sum(1 for e in cache.values() if len(e.get('data', {}).get('trades', [])) == 0)
        non_empty_count = len(cache) - empty_count

        print(f"\n  Summary:")
        print(f"    Total entries: {len(cache)}")
        print(f"    Empty (0 trades): {empty_count}")
        print(f"    Non-empty: {non_empty_count}")

        if empty_count == len(cache):
            print("\n  ⚠️  ALL cached entries are empty!")
            print("  This suggests either:")
            print("    1. These members genuinely have no recent trades (NORMAL)")
            print("    2. API token issue (check Test 2 results)")
            print("    3. API rate limiting")

        return cache

    except json.JSONDecodeError as e:
        print(f"  ❌ Cache file is corrupted: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error reading cache: {e}")
        return None


def test_full_integration():
    """Test 5: Test full integration with overlay.py."""
    print("\n" + "="*70)
    print("TEST 5: Full Integration Test (overlay.py)")
    print("="*70)

    try:
        # Import overlay module
        import overlay
        print("  ✅ overlay module imported")

        # Initialize overlay (needed for globals)
        from config import MatcherConfig
        config = MatcherConfig()
        overlay.init(config)
        print("  ✅ overlay.init() called")

        # Test get_quiver_trades
        test_bioguide = "B001288"
        test_first = "Cory"
        test_last = "Booker"

        print(f"\n  Testing get_quiver_trades('{test_first}', '{test_last}', bioguide='{test_bioguide}')...")

        trades, quiver_url = overlay.get_quiver_trades(test_first, test_last, bioguide=test_bioguide, limit=5)

        print(f"  ✅ Function executed successfully")
        print(f"  Returned {len(trades)} trades")
        print(f"  Quiver URL: {quiver_url}")

        if trades:
            print("\n  ✅ SUCCESS! Trades retrieved from Quiver API:")
            for i, trade in enumerate(trades[:3], 1):
                print(f"    {i}. {trade.get('traded')}: {trade.get('ticker')} - {trade.get('transaction')}")
        else:
            print("\n  ⚠️  No trades returned")
            print("  This is normal if the member has no recent trades.")

        return True

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_known_traders():
    """Test 6: Test with known active traders."""
    print("\n" + "="*70)
    print("TEST 6: Testing with Known Active Traders")
    print("="*70)

    # List of congress members known for active trading
    known_traders = [
        ("Nancy Pelosi", "P000197", "House Speaker known for trades"),
        ("Tommy Tuberville", "T000278", "Senator with frequent trading"),
        ("Josh Gottheimer", "G000583", "House member with active trading"),
        ("Ro Khanna", "K000389", "House member with tech stock trades"),
    ]

    if not REQUESTS_OK:
        print("  ❌ requests library not available")
        return

    try:
        from quiver_api import fetch_congress_trading, load_token
        token = load_token()
    except Exception as e:
        print(f"  ❌ Could not load token: {e}")
        return

    results = []

    for name, bioguide, description in known_traders:
        print(f"\n  Testing: {name} ({bioguide})")
        print(f"  Note: {description}")

        try:
            trades = fetch_congress_trading(bioguide, token=token, timeout=10.0)
            count = len(trades) if isinstance(trades, list) else 0
            results.append((name, bioguide, count, True))

            if count > 0:
                print(f"  ✅ Found {count} trades!")
            else:
                print(f"  ⚠️  No trades (may have no recent activity)")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((name, bioguide, 0, False))

    # Summary
    print("\n" + "="*70)
    print("  Summary of Known Traders Test:")
    print("="*70)

    successful = sum(1 for r in results if r[3])
    with_trades = sum(1 for r in results if r[2] > 0)

    print(f"\n  API calls successful: {successful}/{len(results)}")
    print(f"  Members with trades: {with_trades}/{len(results)}")

    if with_trades > 0:
        print("\n  ✅ API IS WORKING! Found trades for:")
        for name, bioguide, count, success in results:
            if count > 0:
                print(f"    - {name}: {count} trades")
    else:
        print("\n  ⚠️  No trades found for any known active traders")
        print("  This may indicate:")
        print("    1. API token issue (check Test 2)")
        print("    2. API subscription limitations")
        print("    3. Temporary API issue")


def main():
    print("\n" + "="*70)
    print("🔍 QUIVER API INTEGRATION DIAGNOSTIC")
    print("="*70)
    print("\nThis script will test the Quiver API integration and diagnose")
    print("why trades might be empty in your cache.\n")

    results = {}

    # Test 1: Token configuration
    token = test_token_configuration()
    results['token'] = token is not None

    if not token:
        print("\n" + "="*70)
        print("❌ DIAGNOSIS: No API token found")
        print("="*70)
        print("\nWithout a token, the Quiver API cannot be called.")
        print("This is why all cached entries show empty trades.")
        print("\nFIX: Add your Quiver API token to one of these locations:")
        print("  - Environment: export QUIVER_API_TOKEN='your-token'")
        print("  - File: echo 'your-token' > 01_data/quiver_token.txt")
        return False

    # Test 2: Direct API call
    api_data = test_api_direct(token)
    results['api_call'] = api_data is not None
    results['api_has_trades'] = isinstance(api_data, list) and len(api_data) > 0

    # Test 3: Module integration
    results['module'] = test_module_integration()

    # Test 4: Cache status
    cache_data = test_cache_status()
    results['cache'] = cache_data is not None

    # Test 5: Full integration
    results['full_integration'] = test_full_integration()

    # Test 6: Known traders
    test_with_known_traders()

    # Final diagnosis
    print("\n" + "="*70)
    print("📊 FINAL DIAGNOSIS")
    print("="*70)

    print("\n  Test Results:")
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    {test}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n  ✅ ALL TESTS PASSED!")
        print("\n  Conclusion:")
        print("    The Quiver API integration is working correctly.")
        if not results.get('api_has_trades'):
            print("    Empty trades in cache are NORMAL - these members haven't traded recently.")
        else:
            print("    Trades are being fetched successfully from the API.")
    else:
        print("\n  ⚠️  SOME TESTS FAILED")
        print("\n  Likely issues:")
        if not results.get('token'):
            print("    - API token not configured")
        if not results.get('api_call'):
            print("    - API authentication or connection issue")
        if not results.get('module'):
            print("    - Module integration issue")
        if not results.get('full_integration'):
            print("    - Full integration issue")

    print("\n" + "="*70)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
