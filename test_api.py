#!/usr/bin/env python3

import requests
import json
import sys

def test_endpoint(url, description):
    try:
        print(f"🧪 Test: {description}")
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: {data}")
        else:
            print(f"   ❌ Error: {response.text}")
        print()
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Connection error: {str(e)}")
        print()

if __name__ == "__main__":
    print("=== Testing API Endpoints ===\n")
    
    # Test base endpoint
    test_endpoint("http://127.0.0.1:8000/", "Root endpoint")
    
    # Test new health endpoint
    test_endpoint("http://127.0.0.1:8000/health", "Health check")
    
    # Test experiments endpoints
    test_endpoint("http://127.0.0.1:8000/experiments", "Experiments list")
    test_endpoint("http://127.0.0.1:8000/experiments/summary", "Experiments summary")
    
    # Test models endpoint
    test_endpoint("http://127.0.0.1:8000/models", "Models list")
    
    print("=== Test Complete ===")
