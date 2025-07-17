#!/usr/bin/env python3
"""
Test script for the new ESG Analysis endpoint
"""
import requests
import json

def test_esg_endpoint():
    """Test the /esg_analysis endpoint with sample property data"""
    
    # API endpoint
    url = "http://localhost:8010/esg_analysis"
    
    # Sample property data matching the PropertyFeatures schema
    test_data = {
        "propertyFeatures": {
            "propertyType": "HOUSE",
            "subtype": "VILLA",
            "province": "Antwerp",
            "locality": "Antwerpen",
            "postCode": "2000",
            "bedroomCount": 4,
            "bathroomCount": 2,
            "toiletCount": 2,
            "roomCount": 8,
            "habitableSurface": 250,
            "facedeCount": 3,
            "buildingConstructionYear": 1995,
            "buildingCondition": "GOOD",
            "kitchenType": "HYPER_EQUIPPED",
            "heatingType": "GAS",
            "floodZoneType": "NON_FLOOD_ZONE",
            "epcScore": "B",
            "hasLivingRoom": True,
            "hasTerrace": True
        },
        "estimatedPrice": 450000.0,
        "analysis_depth": "detailed"
    }
    
    print("Testing ESG Analysis endpoint...")
    print(f"URL: {url}")
    print(f"Test data: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS! ESG Analysis endpoint working correctly.")
            print(f"Analysis Points: {len(result.get('analysis_points', []))}")
            print(f"ESG Scores: {result.get('esg_scores', {})}")
            print(f"Recommendations: {len(result.get('recommendations', []))}")
            print(f"Compliance Status: {result.get('compliance_status', {})}")
            print(f"Full Report Length: {len(result.get('full_report', ''))}")
            return True
        else:
            print(f"\n❌ FAILED! Status Code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n⚠️  CONNECTION ERROR: Make sure the FastAPI server is running on localhost:8000")
        print("   Run: cd app/backend-api-llm-v2 && python -m uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    test_esg_endpoint()
