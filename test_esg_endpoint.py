#!/usr/bin/env python3
"""
Test script for ESG Analysis endpoint
"""

import requests
import json

# Test payload similar to what the frontend sends
test_payload = {
    "propertyType": "House",
    "subtype": "Villa",
    "province": "Brussels",
    "locality": "Ixelles",
    "postCode": "1050",
    "constructionYear": 1985,
    "surface": 150.0,
    "condition": "Good",
    "epcScore": "D",
    "heatingType": "Gas",
    "estimatedPrice": 450000.0,
    "userProfile": {
        "name": "User",
        "type": "individual",
        "objectives": ["esg_analysis", "energy_efficiency"],
        "language": "en"
    }
}

def test_esg_endpoint():
    url = "http://127.0.0.1:8010/esg-analysis"
    
    print("🧪 Testing ESG Analysis endpoint...")
    print(f"📡 URL: {url}")
    print(f"📋 Payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        response = requests.post(url, json=test_payload, timeout=30)
        
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("📊 ESG Analysis Response:")
            print(json.dumps(result, indent=2))
            
            if "comments" in result and result["comments"]:
                print(f"\n💬 Analysis Points ({len(result['comments'])}):")
                for i, comment in enumerate(result["comments"], 1):
                    print(f"  {i}. {comment}")
            
            print("\n🎉 ESG Analysis endpoint working correctly!")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    test_esg_endpoint()
