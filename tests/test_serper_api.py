"""Test SerpApi key configuration."""

import asyncio
import httpx
from config.settings import get_settings


async def test_serpapi():
    """Test if SerpApi is configured correctly."""
    settings = get_settings()

    if not settings.api.serper_api_key:
        print("❌ SerpApi key not configured")
        print("Set SERPER_API_KEY environment variable")
        return False

    api_key = settings.api.serper_api_key.get_secret_value()
    print(f"✓ SerpApi key found (length: {len(api_key)})")
    print(f"  Testing with serpapi.com...")

    # Test API call using SerpApi format
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # SerpApi uses query parameters, not headers
            params = {
                "api_key": api_key,
                "engine": "google",
                "q": "artificial intelligence trends",
                "num": 5,
            }

            response = await client.get("https://serpapi.com/search", params=params)

            print(f"API Response Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"✓ API test successful!")
                organic_results = result.get("organic_results", [])
                print(f"  Found {len(organic_results)} organic results")
                if organic_results:
                    print(f"  First result: {organic_results[0].get('title', 'N/A')}")
                return True
            elif response.status_code == 401:
                print(f"❌ 401 Unauthorized - API key is invalid")
                print(
                    "  Visit https://serpapi.com/manage-api-key to check your API key"
                )
                return False
            else:
                print(f"❌ API test failed: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return False

    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_serpapi())
    exit(0 if result else 1)
