#!/usr/bin/env python3
"""
Test script to verify ESG scoring alignment between Quick Assessment and Detailed Analysis
"""

import requests
import json
import time

# API URLs (adjust for your environment)
QUICK_ESG_URL = "http://127.0.0.1:8010/esg_quick_analysis"
DETAILED_ESG_URL = "http://127.0.0.1:8010/esg_analysis"

# Test property data
test_property = {
    "propertyFeatures": {
        "propertyType": "HOUSE",
        "subtype": "HOUSE",
        "locality": "Antwerpen",
        "province": "Antwerp",
        "postCode": "2000",
        "habitableSurface": 110,
        "bedroomCount": 3,
        "bathroomCount": 1,
        "toiletCount": 2,
        "buildingConstructionYear": 2000,
        "buildingCondition": "GOOD",
        "kitchenType": "HYPER_EQUIPPED",
        "heatingType": "ELECTRIC",
        "floodZoneType": "NON_FLOOD_ZONE",
        "epcScore": "C",
        "hasLivingRoom": True,
        "hasTerrace": True
    },
    "estimatedPrice": 350000,
    "analysis_depth": "detailed"
}

def test_esg_alignment():
    print("🔍 Testing ESG Scoring Alignment")
    print("=" * 50)
    
    try:
        # Test Quick Analysis
        print("📊 Calling Quick ESG Analysis...")
        quick_response = requests.post(QUICK_ESG_URL, json=test_property, timeout=30)
        quick_response.raise_for_status()
        quick_data = quick_response.json()
        
        print("✅ Quick Analysis Response:")
        print(f"   Environmental: {quick_data['esg_scores']['environmental']}/10")
        print(f"   Social: {quick_data['esg_scores']['social']}/10")
        print(f"   Governance: {quick_data['esg_scores']['governance']}/10")
        print(f"   Overall: {quick_data['esg_scores']['overall']}/10")
        print()
        
        # Test Detailed Analysis
        print("📋 Calling Detailed ESG Analysis...")
        detailed_response = requests.post(DETAILED_ESG_URL, json=test_property, timeout=30)
        detailed_response.raise_for_status()
        detailed_data = detailed_response.json()
        
        print("✅ Detailed Analysis Response:")
        print(f"   Environmental: {detailed_data['esg_scores']['environmental']}/10")
        print(f"   Social: {detailed_data['esg_scores']['social']}/10")
        print(f"   Governance: {detailed_data['esg_scores']['governance']}/10")
        print(f"   Overall: {detailed_data['esg_scores']['overall']}/10")
        print()
        
        # Compare scores
        print("🔄 Score Comparison:")
        env_diff = abs(quick_data['esg_scores']['environmental'] - detailed_data['esg_scores']['environmental'])
        social_diff = abs(quick_data['esg_scores']['social'] - detailed_data['esg_scores']['social'])
        gov_diff = abs(quick_data['esg_scores']['governance'] - detailed_data['esg_scores']['governance'])
        overall_diff = abs(quick_data['esg_scores']['overall'] - detailed_data['esg_scores']['overall'])
        
        print(f"   Environmental difference: {env_diff:.1f}")
        print(f"   Social difference: {social_diff:.1f}")
        print(f"   Governance difference: {gov_diff:.1f}")
        print(f"   Overall difference: {overall_diff:.1f}")
        print()
        
        # Determine alignment
        max_acceptable_diff = 1.5  # Allow up to 1.5 point difference
        
        if (env_diff <= max_acceptable_diff and 
            social_diff <= max_acceptable_diff and 
            gov_diff <= max_acceptable_diff and 
            overall_diff <= max_acceptable_diff):
            print("✅ ALIGNMENT SUCCESS: Scores are consistent!")
            print("   Users will see coherent ESG assessments.")
        else:
            print("⚠️  ALIGNMENT ISSUE: Significant score differences detected.")
            print("   This may confuse users comparing Quick vs Detailed analysis.")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        print("💡 Make sure both backend servers are running:")
        print("   cd app/backend-api-llm-v2")
        print("   python -m uvicorn main:app --reload --port 8010")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_user_experience_flow():
    print("\n🎯 Testing User Experience Flow")
    print("=" * 50)
    
    # Test different property types to ensure consistency
    test_properties = [
        {
            "name": "Energy Efficient House",
            "epcScore": "A",
            "heatingType": "ELECTRIC",
            "locality": "Brussels"
        },
        {
            "name": "Old House Needing Renovation",
            "epcScore": "F",
            "heatingType": "GAS",
            "locality": "Antwerpen"
        },
        {
            "name": "Modern Apartment",
            "epcScore": "B",
            "heatingType": "ELECTRIC",
            "locality": "Gent"
        }
    ]
    
    for prop in test_properties:
        print(f"\n🏠 Testing: {prop['name']}")
        print(f"   EPC: {prop['epcScore']}, Heating: {prop['heatingType']}, Location: {prop['locality']}")
        
        test_data = test_property.copy()
        test_data["propertyFeatures"]["epcScore"] = prop["epcScore"]
        test_data["propertyFeatures"]["heatingType"] = prop["heatingType"]
        test_data["propertyFeatures"]["locality"] = prop["locality"]
        
        try:
            # Quick analysis
            quick_response = requests.post(QUICK_ESG_URL, json=test_data, timeout=20)
            if quick_response.status_code == 200:
                quick_overall = quick_response.json()['esg_scores']['overall']
                print(f"   Quick Assessment Score: {quick_overall}/10")
            else:
                print(f"   ❌ Quick Assessment failed: {quick_response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Quick Assessment error: {str(e)[:50]}...")

if __name__ == "__main__":
    print("🚀 ESG Alignment Test Suite")
    print("Testing alignment between ESG Quick Assessment and Detailed Analysis")
    print()
    
    success = test_esg_alignment()
    
    if success:
        test_user_experience_flow()
        print("\n🎉 Test completed!")
        print("\n📋 Next Steps:")
        print("1. Start your React frontend")
        print("2. Make a price prediction")
        print("3. Verify ESG Quick Assessment shows AI-powered scores")
        print("4. Click 'Generate Comprehensive ESG Analysis'")
        print("5. Confirm both panels show similar scores (±1.5 points)")
    else:
        print("\n❌ Test failed - check your backend setup")
