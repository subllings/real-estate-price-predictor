#!/usr/bin/env python3
"""
Test script to validate ESG display fixes
Simulates the user flow and validates the changes made
"""

import requests
import json
import time

def test_esg_display_fixes():
    """Test the ESG display fixes"""
    
    print("🧪 Testing ESG Display Fixes")
    print("=" * 50)
    
    # Test data - typical Belgian property
    test_property = {
        "propertyFeatures": {
            "propertyType": "HOUSE",
            "subtype": "VILLA",
            "locality": "Brussels",
            "province": "Brussels",
            "postCode": "1000",
            "habitableSurface": 150,
            "buildingConstructionYear": 2018,
            "epcScore": "B",
            "heatingType": "GAS",
            "buildingCondition": "GOOD",
            "bedroomCount": 3,
            "bathroomCount": 2,
            "toiletCount": 2,
            "hasLivingRoom": True,
            "hasTerrace": True,
            "floodZoneType": "NON_FLOOD_ZONE"
        },
        "estimatedPrice": 450000,
        "analysis_depth": "detailed"
    }
    
    print("1. Testing ESG Analysis Endpoint...")
    
    try:
        # Test the ESG API endpoint
        response = requests.post(
            "http://localhost:8010/esg_analysis",
            json=test_property,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ ESG API responds successfully")
            
            # Validate English content
            analysis_text = " ".join(data.get("analysis_points", []))
            french_words = ["énergie", "efficacité", "belgique", "performance", "rénovation"]
            
            contains_french = any(word in analysis_text.lower() for word in french_words)
            if not contains_french:
                print("   ✅ Content is in English (no French words detected)")
            else:
                print("   ⚠️  French content still detected in analysis")
            
            # Validate emoji removal
            emoji_chars = ["🤖", "⏳", "📊", "🔍", "💡", "✅", "🎯", "🌱", "👥", "⚖️", "🏆"]
            contains_emojis = any(emoji in analysis_text for emoji in emoji_chars)
            
            if not contains_emojis:
                print("   ✅ No emojis found in analysis content")
            else:
                print("   ⚠️  Emojis still present in analysis content")
            
            # Validate scores structure
            if "esg_scores" in data:
                scores = data["esg_scores"]
                expected_keys = ["environmental", "social", "governance", "overall"]
                if all(key in scores for key in expected_keys):
                    print("   ✅ ESG scores structure is correct")
                else:
                    print("   ⚠️  Missing expected score categories")
            
        else:
            print(f"   ❌ ESG API error: {response.status_code}")
            print(f"      Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("   ⚠️  Backend not running - start with:")
        print("      cd app/backend-api-llm-v2 && python -m uvicorn main:app --reload --port 8010")
    except Exception as e:
        print(f"   ❌ Error testing ESG API: {e}")
    
    print("\n2. Frontend Component Validation...")
    
    # Check frontend files for fixes
    frontend_files = [
        "app/frontend-react/src/components/ESGPanel/ESGPanel.jsx",
        "app/frontend-react/src/components/PropertyForm/PropertyForm.js",
        "app/frontend-react/src/components/EsgSummary/EsgSummary.jsx"
    ]
    
    for file_path in frontend_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for emoji removal
                emoji_chars = ["🤖", "⏳", "📊", "🔍", "💡", "✅", "🎯", "🌱", "👥", "⚖️", "🏆"]
                has_emojis = any(emoji in content for emoji in emoji_chars)
                
                # Check for French text removal
                french_phrases = [
                    "Génération ESG en cours",
                    "Analyse en cours",
                    "Agent LLM Azure OpenAI actif",
                    "insights générés",
                    "Cette analyse est basée sur l'IA"
                ]
                has_french = any(phrase in content for phrase in french_phrases)
                
                file_name = file_path.split('/')[-1]
                if not has_emojis and not has_french:
                    print(f"   ✅ {file_name}: Emojis and French text removed")
                else:
                    if has_emojis:
                        print(f"   ⚠️  {file_name}: Still contains emojis")
                    if has_french:
                        print(f"   ⚠️  {file_name}: Still contains French text")
                
        except FileNotFoundError:
            print(f"   ❌ File not found: {file_path}")
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")
    
    print("\n3. ESG Summary vs Analysis Report Distinction...")
    
    try:
        with open("app/frontend-react/src/components/EsgSummary/EsgSummary.jsx", 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for improved labeling
            improvements = [
                ("ESG Quick Assessment", "Title updated to indicate preliminary nature"),
                ("Preliminary Score", "Score labeled as preliminary"),
                ("Generate Comprehensive ESG Analysis", "Button text clarifies what will happen"),
                ("preliminary assessment based on property features", "Explanation added")
            ]
            
            for check, description in improvements:
                if check in content:
                    print(f"   ✅ {description}")
                else:
                    print(f"   ⚠️  Missing: {description}")
                    
    except Exception as e:
        print(f"   ❌ Error validating ESG Summary: {e}")
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("\nTo test manually:")
    print("1. Start backend: cd app/backend-api-llm-v2 && python -m uvicorn main:app --reload --port 8010")
    print("2. Start React frontend")
    print("3. Make a price prediction")
    print("4. Check ESG Quick Assessment (should be in English, no emojis)")
    print("5. Click 'Generate Comprehensive ESG Analysis'")
    print("6. Verify loading animation and final report are in English without emojis")

if __name__ == "__main__":
    test_esg_display_fixes()
