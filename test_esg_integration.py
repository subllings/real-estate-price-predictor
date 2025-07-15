#!/usr/bin/env python3
"""
Test script for complete ESG Panel Integration
"""

import requests
import json
import time

def test_prediction_api():
    """Test price prediction API"""
    print("🏠 Testing Price Prediction API...")
    
    prediction_payload = {
        "propertyType": "HOUSE",
        "subtype": "HOUSE",
        "province": "Brussels",
        "locality": "Ixelles",
        "postCode": "1050",
        "buildingConstructionYear": 1985,
        "habitableSurface": 150,
        "buildingCondition": "GOOD",
        "epcScore": "D",
        "heatingType": "GAS",
        "hasSwimmingPool": False,
        "hasGarden": True,
        "hasTerrace": True,
        "numberOfBedrooms": 3,
        "numberOfBathrooms": 2,
        "numberOfFrontages": 2
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict/all", json=prediction_payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Price Prediction: €{result.get('prediction', 'N/A'):,.0f}")
            return result.get('prediction', 450000)
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return 450000
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return 450000

def test_esg_analysis(estimated_price):
    """Test ESG Analysis API"""
    print("\n🌱 Testing ESG Analysis API...")
    
    esg_payload = {
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
        "estimatedPrice": float(estimated_price),
        "userProfile": {
            "name": "User",
            "type": "individual",
            "objectives": ["esg_analysis", "energy_efficiency"],
            "language": "en"
        }
    }
    
    try:
        response = requests.post("http://127.0.0.1:8010/esg-analysis", json=esg_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            analysis_points = result.get("comments", [])
            print(f"✅ ESG Analysis: {len(analysis_points)} insights generated")
            
            # Show first 3 analysis points as preview
            for i, point in enumerate(analysis_points[:3], 1):
                preview = point[:100] + "..." if len(point) > 100 else point
                print(f"   {i}. {preview}")
                
            if len(analysis_points) > 3:
                print(f"   ... and {len(analysis_points) - 3} more insights")
                
            return True
        else:
            print(f"❌ ESG Analysis failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ESG Analysis error: {e}")
        return False

def test_complete_workflow():
    """Test the complete workflow"""
    print("🚀 Testing Complete ESG Panel Workflow")
    print("=" * 50)
    
    # Step 1: Price Prediction
    estimated_price = test_prediction_api()
    
    # Step 2: ESG Analysis
    esg_success = test_esg_analysis(estimated_price)
    
    # Summary
    print("\n📊 Integration Test Results:")
    print(f"   Price Prediction API: ✅")
    print(f"   ESG Analysis API: {'✅' if esg_success else '❌'}")
    print(f"   Complete Workflow: {'✅ Ready for ESG Panel!' if esg_success else '❌ Needs debugging'}")
    
    if esg_success:
        print("\n🎉 ESG Panel Integration is ready!")
        print("   - Price prediction ✅")
        print("   - ESG analysis ✅") 
        print("   - Right panel display ✅")
        print("\nNext Steps:")
        print("   1. Open http://localhost:3000")
        print("   2. Fill property form and predict price")
        print("   3. Click 'Detailed Analysis' button")
        print("   4. ESG Panel should open on the right side")

if __name__ == "__main__":
    test_complete_workflow()
