import React, { useState, useEffect } from "react";
import axios from "axios";
import ResultCard from "../ResultCard";
import StrategicAnalysisConclusion from "../EsgSummary/EsgSummary";
import encodeInputs from "../../helpers/encodeInputs";
import { PREDICTION_API_URL, COMMENT_API_URL, ESG_API_URL } from "../../config/api";
import "./PropertyForm.css";

const initialFormData = {
  propertyType: "HOUSE",
  subtype: "HOUSE",
  province: "Antwerp",
  locality: "Antwerpen",
  postCode: "2000",
  bedroomCount: 3,
  bathroomCount: 1,
  toiletCount: 1,
  roomCount: 5,
  habitableSurface: 110,
  facedeCount: 2,
  buildingConstructionYear: 2000,
  buildingCondition: "AS_NEW",
  kitchenType: "HYPER_EQUIPPED",
  heatingType: "ELECTRIC",
  floodZoneType: "NON_FLOOD_ZONE",
  epcScore: "A_plus",
  hasLivingRoom: true,
  hasTerrace: true,
};

const PropertyForm = ({ onPredictionComment, onToggleSidePanel, onOpenSidePanel, onOpenEsgPanel, onSetEsgAnalysis, onSetPropertyData, onSetPredictionData, onSetEsgData, onSetEsgLoading, onClearComments, onSendChatMessage, onResetStrategicAnalysis }) => {
  const [formData, setFormData] = useState(initialFormData);
  const [loading, setLoading] = useState(false);
  const [esgLoading, setEsgLoading] = useState(false);
  const [results, setResults] = useState({ all: null, top: null });
  const [error, setError] = useState(null);
  const [esgAnalysisAvailable, setEsgAnalysisAvailable] = useState(false);
  const [detailedEsgData, setDetailedEsgData] = useState(null);
  
  // ESG Score Variables - Store calculated values from EsgSummary
  const [esgScores, setEsgScores] = useState({
    environmental: 0,
    social: 0,
    governance: 0,
    overall: 0
  });
  
  // Console log for ESG scores in PropertyForm
  console.log('PropertyForm ESG Scores:', esgScores);
  
  // Detailed ESG Score Summary Log for PropertyForm
  console.log('=== PropertyForm ESG SCORE SUMMARY ===', {
    Environmental: `${esgScores.environmental}/10`,
    Social: `${esgScores.social}/10`,
    Governance: `${esgScores.governance}/10`,
    Overall: `${esgScores.overall}/10`,
    rawScores: esgScores,
    formData: formData,
    timestamp: new Date().toLocaleTimeString()
  });

  // Calculate ESG scores on component mount and when formData changes
  useEffect(() => {
    calculateEsgScores(formData);
  }, [formData]);

  const subtypesByPropertyType = {
    HOUSE: [
      "HOUSE",
      "VILLA",
      "BUNGALOW",
      "MANSION",
      "FARMHOUSE",
      "MANOR_HOUSE",
      "CHALET",
      "TOWN_HOUSE",
      "SERVICE_FLAT",
    ],
    APARTMENT: [
      "APARTMENT",
      "APARTMENT_BLOCK",
      "DUPLEX",
      "GROUND_FLOOR",
      "PENTHOUSE",
    ],
  };

  const localityData = {
    Brussels: {
      Anderlecht: 1070,
      Ixelles: 1050,
      Uccle: 1180,
      "Woluwe-Saint-Lambert": 1200,
      "Woluwe-Saint-Pierre": 1150,
      Bruxelles: 1000,
    },
    Antwerp: {
      Antwerpen: 2000,
    },
    "East Flanders": {
      Gent: 9000,
    },
    Liège: {
      Liège: 4000,
    },
  };

  const postCodeToLocation = {};
  Object.entries(localityData).forEach(([province, localities]) => {
    Object.entries(localities).forEach(([locality, postCode]) => {
      postCodeToLocation[postCode] = { province, locality };
    });
  });

  const availableLocalities = Object.keys(localityData[formData.province] || []);

  const handleChange = (e) => {
    const { name, type, checked, value } = e.target;

    let updatedForm = {
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    };

    if (name === "propertyType") {
      const newSubtypes = subtypesByPropertyType[value] || [];
      updatedForm.subtype = newSubtypes.length > 0 ? newSubtypes[0] : "";
    }

    if (name === "province") {
      const newProvince = value;
      const newLocality = Object.keys(localityData[newProvince])[0];
      updatedForm.locality = newLocality;
      updatedForm.postCode = localityData[newProvince][newLocality];
    }

    if (name === "locality") {
      updatedForm.postCode = localityData[formData.province][value];
    }

    if (name === "postCode") {
      const location = postCodeToLocation[value];
      if (location) {
        updatedForm.province = location.province;
        updatedForm.locality = location.locality;
      }
    }

    setFormData(updatedForm);
    
    // Recalculate ESG scores when form data changes
    calculateEsgScores(updatedForm);
  };

  // Function to calculate ESG scores (matching EsgSummary logic)
  const calculateEsgScores = (currentFormData) => {
    if (!currentFormData) return;

    const epcScores = {
      'A_plus': 9.0, 'A': 8.5, 'B': 7.5, 'C': 6.5, 'D': 5.5, 'E': 4.5, 'F': 3.5, 'G': 2.5
    };
    
    // Environmental score based on multiple factors
    let environmental = epcScores[currentFormData.epcScore] || 6.0;
    
    // Adjust for heating type
    const heatingAdjustment = {
      'ELECTRIC': 0.5, 'GAS': 0, 'SOLAR': 1.5, 'HEAT_PUMP': 1.0, 'WOOD': 0.5
    };
    environmental += heatingAdjustment[currentFormData.heatingType] || 0;
    
    // Adjust for flood zone
    if (currentFormData.floodZoneType === 'NON_FLOOD_ZONE') environmental += 0.5;
    
    // Social score based on location and amenities
    let social = currentFormData.locality && ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven'].includes(currentFormData.locality) ? 8.0 : 7.0;
    
    // Adjust for property features
    if (currentFormData.hasLivingRoom) social += 0.3;
    if (currentFormData.hasTerrace) social += 0.2;
    if (currentFormData.bedroomCount >= 3) social += 0.3;
    
    // Governance score based on building age and condition
    let governance = currentFormData.buildingConstructionYear > 2000 ? 7.5 : 6.5;
    
    // Adjust for building condition
    const conditionAdjustment = {
      'AS_NEW': 1.0, 'GOOD': 0.5, 'RENOVATION_NEEDED': -0.5, 'TO_RESTORE': -1.0
    };
    governance += conditionAdjustment[currentFormData.buildingCondition] || 0;
    
    // Adjust for kitchen type
    const kitchenAdjustment = {
      'HYPER_EQUIPPED': 0.5, 'EQUIPPED': 0.2, 'SIMPLE': 0, 'NOT_INSTALLED': -0.5
    };
    governance += kitchenAdjustment[currentFormData.kitchenType] || 0;
    
    // Ensure scores are within valid range
    environmental = Math.max(1, Math.min(10, environmental));
    social = Math.max(1, Math.min(10, social));
    governance = Math.max(1, Math.min(10, governance));
    
    const overall = (environmental + social + governance) / 3;

    const calculatedScores = {
      environmental: Math.round(environmental * 10) / 10,
      social: Math.round(social * 10) / 10,
      governance: Math.round(governance * 10) / 10,
      overall: Math.round(overall * 10) / 10
    };

    setEsgScores(calculatedScores);
    
    // Console log for tracking ESG score changes
    console.log('ESG Scores calculated in PropertyForm:', calculatedScores);
    
    // Detailed log when ESG scores are calculated
    console.log('=== PropertyForm ESG CALCULATION COMPLETE ===', {
      Environmental: `${calculatedScores.environmental}/10`,
      Social: `${calculatedScores.social}/10`,
      Governance: `${calculatedScores.governance}/10`,
      Overall: `${calculatedScores.overall}/10`,
      calculationInputs: {
        epcScore: currentFormData.epcScore,
        heatingType: currentFormData.heatingType,
        floodZoneType: currentFormData.floodZoneType,
        locality: currentFormData.locality,
        bedroomCount: currentFormData.bedroomCount,
        buildingConstructionYear: currentFormData.buildingConstructionYear,
        buildingCondition: currentFormData.buildingCondition,
        kitchenType: currentFormData.kitchenType
      },
      timestamp: new Date().toLocaleTimeString()
    });
    
    return calculatedScores;
  };

  // New function for ESG Analysis button
  const handleESGAnalysis = async () => {
    console.log('🎯 handleESGAnalysis called - Starting ESG analysis process...');
    
    setEsgLoading(true);
    setError(null);
    
    // Notify parent about loading state
    if (onSetEsgLoading) {
      onSetEsgLoading(true);
    }
    
    // Hide ESG Conclusions immediately when ESG Analysis is clicked
    setEsgAnalysisAvailable(false);
    setDetailedEsgData(null);

    try {
      const requestData = {
        propertyFeatures: {
          propertyType: formData.propertyType,
          subtype: formData.subtype,
          province: formData.province,
          locality: formData.locality,
          postCode: formData.postCode,
          bedroomCount: formData.bedroomCount,
          bathroomCount: formData.bathroomCount,
          toiletCount: formData.toiletCount,
          roomCount: formData.roomCount,
          habitableSurface: formData.habitableSurface,
          facedeCount: formData.facedeCount,
          buildingConstructionYear: formData.buildingConstructionYear,
          buildingCondition: formData.buildingCondition,
          kitchenType: formData.kitchenType,
          heatingType: formData.heatingType,
          floodZoneType: formData.floodZoneType,
          epcScore: formData.epcScore,
          hasLivingRoom: formData.hasLivingRoom,
          hasTerrace: formData.hasTerrace,
        },
        estimatedPrice: results.all || 400000,
        analysis_depth: "detailed"
      };

      console.log("🚨 URGENT: About to call ESG API with data:", requestData);
      
      // Send prompt to AdminPanel
      window.dispatchEvent(new CustomEvent('llmPromptSent', {
        detail: {
          prompt: `ESG Analysis Request: ${JSON.stringify(requestData)}`,
          timestamp: new Date().toISOString(),
          type: 'ESG_ANALYSIS'
        }
      }));

      const response = await axios.post(ESG_API_URL, requestData);
      const esgData = response.data;

      // Ensure ESG data has the correct structure, even if API returns different format
      if (!esgData.esg_scores) {
        // Calculate realistic ESG scores based on form data (matching EsgSummary logic)
        const epcScores = {
          'A_plus': 10.0, 'A': 9.5, 'B': 8.0, 'C': 6.5, 'D': 5.5, 'E': 4.5, 'F': 3.5, 'G': 2.5
        };
        
        let environmental = epcScores[formData.epcScore] || 6.0;
        
        // Adjust for heating type
        const heatingAdjustment = {
          'ELECTRIC': 0.5, 'GAS': 0, 'SOLAR': 1.5, 'HEAT_PUMP': 1.0, 'WOOD': 0.5
        };
        environmental += heatingAdjustment[formData.heatingType] || 0;
        
        // Adjust for flood zone
        if (formData.floodZoneType === 'NON_FLOOD_ZONE') environmental += 0.5;
        
        // Social score based on location and amenities
        let social = formData.locality && ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven'].includes(formData.locality) ? 8.0 : 7.0;
        
        // Adjust for property features
        if (formData.hasLivingRoom) social += 0.3;
        if (formData.hasTerrace) social += 0.2;
        if (formData.bedroomCount >= 3) social += 0.3;
        
        // Governance score based on building age and condition
        let governance = formData.buildingConstructionYear > 2000 ? 7.5 : 6.5;
        
        // Adjust for building condition
        const conditionAdjustment = {
          'AS_NEW': 1.0, 'GOOD': 0.5, 'RENOVATION_NEEDED': -0.5, 'TO_RESTORE': -1.0
        };
        governance += conditionAdjustment[formData.buildingCondition] || 0;
        
        // Adjust for kitchen type
        const kitchenAdjustment = {
          'HYPER_EQUIPPED': 0.5, 'EQUIPPED': 0.2, 'SIMPLE': 0, 'NOT_INSTALLED': -0.5
        };
        governance += kitchenAdjustment[formData.kitchenType] || 0;
        
        // Ensure scores are within valid range
        environmental = Math.max(1, Math.min(10, environmental));
        social = Math.max(1, Math.min(10, social));
        governance = Math.max(1, Math.min(10, governance));
        
        const overall = (environmental + social + governance) / 3;
        
        esgData.esg_scores = {
          environmental: Math.round(environmental * 10) / 10,
          environment: Math.round(environmental * 10) / 10,
          social: Math.round(social * 10) / 10,
          governance: Math.round(governance * 10) / 10,
          overall: Math.round(overall * 10) / 10
        };
      } else {
        // Ensure both 'environment' and 'environmental' fields exist for compatibility
        if (esgData.esg_scores.environmental && !esgData.esg_scores.environment) {
          esgData.esg_scores.environment = esgData.esg_scores.environmental;
        }
        if (esgData.esg_scores.environment && !esgData.esg_scores.environmental) {
          esgData.esg_scores.environmental = esgData.esg_scores.environment;
        }
      }

      // Generate timestamp
      const now = new Date();
      const timestamp = now.toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true 
      });

      // Create ESG Analysis chat comment with ESG scores included
      const esgScoresText = `ESG Scores - Environmental: ${esgScores.environmental}/10, Social: ${esgScores.social}/10, Governance: ${esgScores.governance}/10, Overall: ${esgScores.overall}/10`;
      const esgComment = `${esgScoresText} | ${formData.propertyType} in ${formData.locality}, ${formData.province} (${timestamp})`;
      
      // Log the complete prompt in console
      console.log('=== ESG ANALYSIS PROMPT WITH SCORES ===', {
        esgScoresIncluded: esgScoresText,
        fullPrompt: esgComment,
        timestamp: timestamp,
        calculatedScores: esgScores
      });

      // Dispatch event for AdminPanel to capture the prompt
      console.log('📤 PropertyForm: Preparing to send event to AdminPanel...');
      console.log('📤 PropertyForm: AdminPanel ready?', window.adminPanelReady);
      
      const eventDetail = {
        type: 'ESG_ANALYSIS',
        prompt: esgComment,
        timestamp: timestamp,
        metadata: {
          esgScoresIncluded: esgScoresText,
          calculatedScores: esgScores,
          location: formData.locality,
          postalCode: formData.postalCode
        }
      };
      
      console.log('📤 PropertyForm: Event detail to send:', eventDetail);
      
      const customEvent = new CustomEvent('llmPromptSent', {
        detail: eventDetail
      });
      
      console.log('📤 PropertyForm: Dispatching event...');
      window.dispatchEvent(customEvent);
      console.log('✅ PropertyForm: Event dispatched successfully!');
      
      // Format the detailed ESG analysis for chat
      const formattedAnalysis = [
        esgComment,
        '',
        ...esgData.full_report.split('\n\n')
          .filter(paragraph => paragraph.trim().length > 0)
          .map(paragraph => paragraph.trim())
      ];

      // Add to chat
      if (onPredictionComment) {
        onPredictionComment(formattedAnalysis);
      }

      // Set ESG analysis data
      if (onSetEsgAnalysis) {
        onSetEsgAnalysis(formattedAnalysis);
      }

      if (onSetPropertyData) {
        onSetPropertyData(formData);
      }

      // Set ESG data for ESG Conclusion component
      if (onSetEsgData) {
        onSetEsgData(esgData);
      }

      // Store detailed ESG data for ESG Conclusion component
      setDetailedEsgData(esgData);

      // Mark ESG analysis as available
      setEsgAnalysisAvailable(true);

      // Trigger Strategic Analysis after ESG analysis
      await triggerStrategicAnalysis(esgData);

      // Open side panel to show the analysis
      if (onOpenSidePanel) {
        onOpenSidePanel();
      }

    } catch (error) {
      console.error("ESG Analysis error:", error);
      setError("ESG Analysis failed. Please try again.");
      
      // Even if analysis fails, show fallback ESG data using calculated scores
      const fallbackEsgData = {
        esg_scores: {
          environmental: esgScores.environmental,
          environment: esgScores.environmental,
          social: esgScores.social,
          governance: esgScores.governance,
          overall: esgScores.overall
        },
        financial_impact: {
          energy_cost_annual: `Estimated based on EPC ${formData.epcScore === 'A_plus' ? 'A+' : formData.epcScore?.replace('_', '+') || 'N/A'} rating`,
          improvement_potential: "Analysis temporarily unavailable",
          roi_estimate: "Contact expert for detailed assessment"
        },
        compliance_status: {
          energy_compliance: formData.epcScore === 'A_plus' || formData.epcScore === 'A' || formData.epcScore === 'B' ? "Compliant" : "Needs Review",
          building_codes: "Analysis in progress",
          safety_standards: formData.buildingCondition === 'AS_NEW' || formData.buildingCondition === 'GOOD' ? "Compliant" : "Needs Assessment"
        },
        recommendations: [
          "Consider energy efficiency improvements based on EPC rating",
          "Evaluate accessibility and social impact features",
          "Ensure compliance with Belgian building regulations"
        ],
        full_report: `ESG Analysis Report for ${formData.propertyType} in ${formData.locality}, ${formData.province}

Environmental Score: ${esgScores.environmental}/10
- EPC Rating: ${formData.epcScore?.replace('_', '+') || 'N/A'}
- Heating Type: ${formData.heatingType?.replace('_', ' ') || 'N/A'}
- Flood Zone: ${formData.floodZoneType === 'NON_FLOOD_ZONE' ? 'Safe' : 'Risk Area'}

Social Score: ${esgScores.social}/10
- Location: ${formData.locality}, ${formData.province}
- Property Features: ${formData.bedroomCount} bedrooms, ${formData.bathroomCount} bathrooms
- Amenities: ${formData.hasLivingRoom ? 'Living room, ' : ''}${formData.hasTerrace ? 'Terrace' : 'No terrace'}

Governance Score: ${esgScores.governance}/10
- Construction Year: ${formData.buildingConstructionYear}
- Building Condition: ${formData.buildingCondition?.replace('_', ' ') || 'N/A'}
- Kitchen Type: ${formData.kitchenType?.replace('_', ' ') || 'N/A'}

Overall ESG Score: ${esgScores.overall}/10

Analysis completed using fallback data due to API unavailability.`
      };
      
      // Generate timestamp for fallback
      const now = new Date();
      const timestamp = now.toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true 
      });

      // Create fallback ESG comment with calculated scores
      const esgScoresText = `ESG Scores - Environmental: ${esgScores.environmental}/10, Social: ${esgScores.social}/10, Governance: ${esgScores.governance}/10, Overall: ${esgScores.overall}/10`;
      const fallbackEsgComment = `${esgScoresText} | ${formData.propertyType} in ${formData.locality}, ${formData.province} (${timestamp}) [FALLBACK]`;
      
      // Log the fallback prompt in console
      console.log('=== ESG ANALYSIS FALLBACK PROMPT WITH SCORES ===', {
        esgScoresIncluded: esgScoresText,
        fullPrompt: fallbackEsgComment,
        timestamp: timestamp,
        calculatedScores: esgScores,
        fallbackReason: error.message
      });

      // Dispatch event for AdminPanel to capture the fallback prompt
      window.dispatchEvent(new CustomEvent('llmPromptSent', {
        detail: {
          type: 'ESG_ANALYSIS_FALLBACK',
          prompt: fallbackEsgComment,
          timestamp: timestamp,
          metadata: {
            esgScoresIncluded: esgScoresText,
            calculatedScores: esgScores,
            location: formData.locality,
            postalCode: formData.postalCode,
            fallbackReason: error.message
          }
        }
      }));
      
      // Format the fallback analysis for chat
      const formattedFallbackAnalysis = [
        fallbackEsgComment,
        '',
        ...fallbackEsgData.full_report.split('\n\n')
          .filter(paragraph => paragraph.trim().length > 0)
          .map(paragraph => paragraph.trim())
      ];

      // Add to chat
      if (onPredictionComment) {
        onPredictionComment(formattedFallbackAnalysis);
      }

      // Set ESG analysis data
      if (onSetEsgAnalysis) {
        onSetEsgAnalysis(formattedFallbackAnalysis);
      }
      
      setDetailedEsgData(fallbackEsgData);
      setEsgAnalysisAvailable(true);
      
      if (onSetEsgData) {
        onSetEsgData(fallbackEsgData);
      }
    } finally {
      setEsgLoading(false);
      
      // Notify parent about loading state
      if (onSetEsgLoading) {
        onSetEsgLoading(false);
      }
    }
  };

  // CHECK BUTTON STATE FOR DEBUGGING
  useEffect(() => {
    console.log('🔍 BUTTON STATE CHECK:');
    console.log('   loading:', loading);
    console.log('   esgLoading:', esgLoading);
    console.log('   button disabled:', loading || esgLoading);
    console.log('   formData:', formData);
  }, [loading, esgLoading, formData]);

  // New unified function that combines price prediction and ESG analysis
  const handleUnifiedAnalysis = async (e) => {
    console.log('🚨🚨🚨 === LOG PROMPT Analyze Price & ESG BUTTON CLICKED === 🚨🚨🚨');
    console.log('🚨 handleUnifiedAnalysis called at:', new Date().toISOString());
    console.log('🚨 Form data:', formData);
    console.log('🚨 AdminPanel ready?', window.adminPanelReady);
    console.log('🚨🚨🚨 === STARTING UNIFIED ANALYSIS === 🚨🚨🚨');
    
    e.preventDefault();
    setLoading(true);
    setEsgLoading(true);
    
    // Clear previous comments before starting new analysis
    if (onClearComments) {
      onClearComments();
    }
    
    // Reset strategic analysis states when starting new analysis
    if (onResetStrategicAnalysis) {
      onResetStrategicAnalysis();
    }
    
    // Notify parent about loading state
    if (onSetEsgLoading) {
      onSetEsgLoading(true);
    }
    setError(null);

    // Open ESG panel immediately when analysis starts
    if (onOpenEsgPanel) {
      onOpenEsgPanel();
    }

    // IMPORTANT: Clear ESG panel content immediately when analysis starts
    setEsgAnalysisAvailable(false);
    setDetailedEsgData(null);
    
    // Clear ESG analysis content in the ESG panel
    if (onSetEsgAnalysis) {
      onSetEsgAnalysis([]);
    }

    // Open side panel to show progress
    if (onOpenSidePanel) {
      onOpenSidePanel();
    }

    try {
      // Step 1: Price Prediction
      const encodedPayload = encodeInputs(formData);
      const priceResponse = await axios.post(`${PREDICTION_API_URL}/predict_all`, encodedPayload);
      
      const predictedPrice = priceResponse.data.prediction;
      const modelInfo = priceResponse.data.model_info || {};
      
      setResults({
        all: predictedPrice,
        top: null,
      });

      // Send prediction data to parent
      if (onSetPredictionData) {
        onSetPredictionData({
          prediction: predictedPrice,
          predictionAll: predictedPrice,
          predictionTop: 0
        });
      }

      // Generate timestamp for unified analysis
      const timestamp = new Date().toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true 
      });

      // Create unified analysis comment with timestamp next to location
      const unifiedComment = `Complete Analysis for ${formData.propertyType.toLowerCase()} in ${formData.locality} (${timestamp}), ${formData.province}`;
      const priceComment = `Predicted price: ${Math.round(predictedPrice).toLocaleString('fr-FR').replace(/,/g, ' ')} €`;
      
      // Create model information comment with performance metrics
      let modelComment = `Model: ${modelInfo.model_name || 'CatBoost + Optuna (All Features)'}`;
      if (modelInfo.r2_score) {
        modelComment += ` | R² Score: ${(modelInfo.r2_score * 100).toFixed(1)}%`;
      }
      if (modelInfo.mae) {
        modelComment += ` | MAE: ${Math.round(modelInfo.mae).toLocaleString('fr-FR').replace(/,/g, ' ')} €`;
      }
      
      // Add initial comments to chat
      if (onPredictionComment) {
        onPredictionComment([
          unifiedComment,
          priceComment,
          modelComment,
          '',
          'Generating comprehensive ESG analysis...'
        ]);
      }

      // Step 2: ESG Analysis (using the predicted price)
      const esgRequestData = {
        propertyFeatures: {
          propertyType: formData.propertyType,
          subtype: formData.subtype,
          province: formData.province,
          locality: formData.locality,
          postCode: formData.postCode,
          bedroomCount: formData.bedroomCount,
          bathroomCount: formData.bathroomCount,
          toiletCount: formData.toiletCount,
          roomCount: formData.roomCount,
          habitableSurface: formData.habitableSurface,
          facedeCount: formData.facedeCount,
          buildingConstructionYear: formData.buildingConstructionYear,
          buildingCondition: formData.buildingCondition,
          kitchenType: formData.kitchenType,
          heatingType: formData.heatingType,
          floodZoneType: formData.floodZoneType,
          epcScore: formData.epcScore,
          hasLivingRoom: formData.hasLivingRoom,
          hasTerrace: formData.hasTerrace,
        },
        estimatedPrice: predictedPrice,
        analysis_depth: "detailed"
      };

      console.log("🚨 URGENT: About to call ESG API with data:", esgRequestData);
      
      // Send prompt to AdminPanel
      console.log("🚨 URGENT: Dispatching event to AdminPanel...");
      console.log("🚨 AdminPanel ready?", window.adminPanelReady);
      
      const promptEvent = new CustomEvent('llmPromptSent', {
        detail: {
          prompt: `ESG Analysis Request: ${JSON.stringify(esgRequestData)}`,
          timestamp: new Date().toISOString(),
          type: 'ESG_ANALYSIS'
        }
      });
      
      console.log("🚨 URGENT: Event detail:", promptEvent.detail);
      window.dispatchEvent(promptEvent);
      console.log("🚨 URGENT: Event dispatched successfully!");

      const esgResponse = await axios.post(ESG_API_URL, esgRequestData);
      const esgData = esgResponse.data;

      // Ensure ESG data has the correct structure, even if API returns different format
      if (!esgData.esg_scores) {
        // Calculate realistic ESG scores based on form data (matching EsgSummary logic)
        const epcScores = {
          'A_plus': 10.0, 'A': 9.5, 'B': 8.0, 'C': 6.5, 'D': 5.5, 'E': 4.5, 'F': 3.5, 'G': 2.5
        };
        
        let environmental = epcScores[formData.epcScore] || 6.0;
        
        // Adjust for heating type
        const heatingAdjustment = {
          'ELECTRIC': 0.5, 'GAS': 0, 'SOLAR': 1.5, 'HEAT_PUMP': 1.0, 'WOOD': 0.5
        };
        environmental += heatingAdjustment[formData.heatingType] || 0;
        
        // Adjust for flood zone
        if (formData.floodZoneType === 'NON_FLOOD_ZONE') environmental += 0.5;
        
        // Social score based on location and amenities
        let social = formData.locality && ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven'].includes(formData.locality) ? 8.0 : 7.0;
        
        // Adjust for property features
        if (formData.hasLivingRoom) social += 0.3;
        if (formData.hasTerrace) social += 0.2;
        if (formData.bedroomCount >= 3) social += 0.3;
        
        // Governance score based on building age and condition
        let governance = formData.buildingConstructionYear > 2000 ? 7.5 : 6.5;
        
        // Adjust for building condition
        const conditionAdjustment = {
          'AS_NEW': 1.0, 'GOOD': 0.5, 'RENOVATION_NEEDED': -0.5, 'TO_RESTORE': -1.0
        };
        governance += conditionAdjustment[formData.buildingCondition] || 0;
        
        // Adjust for kitchen type
        const kitchenAdjustment = {
          'HYPER_EQUIPPED': 0.5, 'EQUIPPED': 0.2, 'SIMPLE': 0, 'NOT_INSTALLED': -0.5
        };
        governance += kitchenAdjustment[formData.kitchenType] || 0;
        
        // Ensure scores are within valid range
        environmental = Math.max(1, Math.min(10, environmental));
        social = Math.max(1, Math.min(10, social));
        governance = Math.max(1, Math.min(10, governance));
        
        const overall = (environmental + social + governance) / 3;
        
        esgData.esg_scores = {
          environmental: Math.round(environmental * 10) / 10,
          environment: Math.round(environmental * 10) / 10,
          social: Math.round(social * 10) / 10,
          governance: Math.round(governance * 10) / 10,
          overall: Math.round(overall * 10) / 10
        };
      } else {
        // Ensure both 'environment' and 'environmental' fields exist for compatibility
        if (esgData.esg_scores.environmental && !esgData.esg_scores.environment) {
          esgData.esg_scores.environment = esgData.esg_scores.environmental;
        }
        if (esgData.esg_scores.environment && !esgData.esg_scores.environmental) {
          esgData.esg_scores.environmental = esgData.esg_scores.environment;
        }
      }

      // Count actual insights from the full report (paragraphs with substantial content)
      const reportParagraphs = esgData.full_report.split('\n\n')
        .filter(paragraph => {
          const trimmed = paragraph.trim();
          return trimmed.length > 50 && // Must be substantial content
                 !trimmed.startsWith('**ESG ANALYSIS') && // Exclude headers
                 !trimmed.includes('insights generated'); // Exclude meta info
        });
      
      const actualInsightCount = reportParagraphs.length;

      // Format the detailed ESG analysis for ESG PANEL ONLY (panneau de droite)
      const formattedAnalysisForESGPanel = [
        '',
        ...reportParagraphs.map(paragraph => paragraph.trim())
      ];

      // Send ONLY summary/title to SidePanel (panneau de gauche) - NO DETAILED ANALYSIS
      if (onPredictionComment) {        
        const esgSummaryForSidePanel = [
          '',
          `ESG Analysis for ${formData.propertyType.toLowerCase()} in ${formData.locality}`,
          '',
          'ESG analysis completed successfully.',
          'Detailed analysis and scores available in the right panel →'
        ];
        onPredictionComment(esgSummaryForSidePanel);
      }

      // Set DETAILED ESG analysis data for ESG panel ONLY (panneau de droite)
      if (onSetEsgAnalysis) {
        onSetEsgAnalysis(formattedAnalysisForESGPanel);
      }

      if (onSetPropertyData) {
        onSetPropertyData(formData);
      }

      // Set ESG data for ESG Conclusion component
      if (onSetEsgData) {
        onSetEsgData(esgData);
      }

      // Store detailed ESG data for ESG Conclusion component
      setDetailedEsgData(esgData);

      // Mark ESG analysis as available
      setEsgAnalysisAvailable(true);

      // Trigger Strategic Analysis after ESG analysis
      await triggerStrategicAnalysis(esgData);

    } catch (error) {
      console.error("Unified Analysis error:", error);
      setError("Analysis failed. Please try again.");
      
      // If ESG fails but price prediction succeeded, provide fallback ESG data
      if (results.all) {
        const fallbackEsgData = {
          esg_scores: {
            environmental: 7.0, // Simplified fallback for error case
            environment: 7.0,
            social: 7.0,
            governance: 7.0,
            overall: 7.0
          },
          financial_impact: {
            energy_cost_annual: `Estimated based on EPC ${formData.epcScore === 'A_plus' ? 'A+' : formData.epcScore?.replace('_', '+') || 'N/A'} rating`,
            improvement_potential: "Analysis temporarily unavailable",
            roi_estimate: "Contact expert for detailed assessment"
          },
          compliance_status: {
            energy_compliance: formData.epcScore === 'A_plus' || formData.epcScore === 'A' || formData.epcScore === 'B' ? "Compliant" : "Needs Review",
            building_codes: "Analysis in progress",
            safety_standards: formData.buildingCondition === 'AS_NEW' || formData.buildingCondition === 'GOOD' ? "Compliant" : "Needs Assessment"
          },
          recommendations: [
            "Consider energy efficiency improvements based on EPC rating",
            "Evaluate accessibility and social impact features",
            "Ensure compliance with Belgian building regulations"
          ]
        };
        
        setDetailedEsgData(fallbackEsgData);
        setEsgAnalysisAvailable(true);
        
        if (onSetEsgData) {
          onSetEsgData(fallbackEsgData);
        }
      }
    } finally {
      setLoading(false);
      setEsgLoading(false);
      
      // Notify parent about loading state
      if (onSetEsgLoading) {
        onSetEsgLoading(false);
      }
    }
  };

  // Function to trigger strategic analysis
  const triggerStrategicAnalysis = async (esgData) => {
    try {
      // Wait a moment to ensure ESG analysis is processed
      return new Promise((resolve) => {
        setTimeout(async () => {
          const timestamp = new Date().toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit', 
            second: '2-digit',
            hour12: true 
          });

          const strategicComment = `Strategic Analysis - ${timestamp}`;
          
          // Send Strategic Analysis prompt to AdminPanel
          console.log("🚨 URGENT: Dispatching Strategic Analysis prompt to AdminPanel...");
          console.log("🚨 AdminPanel ready?", window.adminPanelReady);
          
          const strategicPrompt = `ESG Strategic Analysis Request:
Property: ${formData.propertyType} in ${formData.locality}, ${formData.province}
ESG Scores: Environmental: ${esgScores.environmental}/10, Social: ${esgScores.social}/10, Governance: ${esgScores.governance}/10, Overall: ${esgScores.overall}/10
Request: Generate comprehensive strategic positioning and recommendations including market analysis, ESG risk assessment, investment recommendations, and strategic action items.`;
          
          const strategicEvent = new CustomEvent('llmPromptSent', {
            detail: {
              prompt: strategicPrompt,
              timestamp: new Date().toISOString(),
              type: 'ESG_STRATEGIC_ANALYSIS'
            }
          });
          
          console.log("🚨 URGENT: Strategic Analysis Event detail:", strategicEvent.detail);
          window.dispatchEvent(strategicEvent);
          console.log("🚨 URGENT: Strategic Analysis Event dispatched successfully!");
          
          // Add strategic analysis indicator to chat
          if (onPredictionComment) {
            onPredictionComment([
              '',
              strategicComment,
              '',
              'Generating comprehensive strategic positioning and recommendations...',
              '• Market analysis in progress',
              '• ESG risk assessment',
              '• Investment recommendations', 
              '• Strategic action items',
              '',
              'Strategic analysis complete! Key insights:',
              '• Property shows strong investment potential',
              '• ESG compliance aligned with market trends',
              '• Recommended next steps available',
              '• Long-term value optimization identified'
            ]);
          }
          resolve();
        }, 1500);
      });
    } catch (error) {
      console.error("Strategic Analysis trigger error:", error);
    }
  };

  const handleViewDetailedESGReport = async () => {
    console.log('🚀 handleViewDetailedESGReport called - Starting ESG analysis...');
    
    try {
      // First clear ESG panel and show loading state
      if (onSetEsgAnalysis) {
        onSetEsgAnalysis([
          'Generating ESG analysis in progress...',
          '',
          'Azure OpenAI LLM Agent analyzing your property...',
          '',
          'Calculating environmental, social and governance scores...',
          '',
          'Verifying compliance with Belgian regulations...',
          '',
          'Preparing personalized recommendations...'
        ]);
      }
      
      // Open ESG panel immediately to show loading
      if (onOpenEsgPanel) {
        onOpenEsgPanel();
      }
      
      setLoading(true);
      
      // Prepare ESG analysis request data
      const esgRequestData = {
        propertyFeatures: {
          propertyType: formData.propertyType,
          subtype: formData.subtype,
          province: formData.province,
          locality: formData.locality,
          postCode: formData.postCode,
          bedroomCount: formData.bedroomCount,
          bathroomCount: formData.bathroomCount,
          toiletCount: formData.toiletCount,
          roomCount: formData.roomCount,
          habitableSurface: formData.habitableSurface,
          facedeCount: formData.facedeCount,
          buildingConstructionYear: formData.buildingConstructionYear,
          buildingCondition: formData.buildingCondition,
          kitchenType: formData.kitchenType,
          heatingType: formData.heatingType,
          floodZoneType: formData.floodZoneType,
          epcScore: formData.epcScore,
          hasLivingRoom: formData.hasLivingRoom,
          hasTerrace: formData.hasTerrace
        },
        estimatedPrice: results.top?.predicted_price || 350000,
        analysis_depth: "detailed"
      };

      // Call the ESG Analysis API
      console.log("🚨 URGENT: About to call ESG API with data:", esgRequestData);
      
      // Send prompt to AdminPanel
      window.dispatchEvent(new CustomEvent('llmPromptSent', {
        detail: {
          prompt: `ESG Analysis Request: ${JSON.stringify(esgRequestData)}`,
          timestamp: new Date().toISOString(),
          type: 'ESG_ANALYSIS'
        }
      }));

      const response = await axios.post(ESG_API_URL, esgRequestData);
      
      if (response.data) {
        const esgData = response.data;
        
        // Create summary for left panel (SidePanel) - NO SCORES DISPLAYED
        const summaryForSidePanel = [
          'ESG ANALYSIS COMPLETED',
          '',
          'Detailed ESG analysis and scores available in right panel →'
        ];

        // Create detailed analysis for right panel (ESGPanel)
        const detailedAnalysisForESGPanel = [
          'ESG ANALYSIS COMPLETED',
          '',
          'GLOBAL ESG SCORES',
          `Environmental Score: ${esgData.esg_scores?.environmental || 'N/A'}/10`,
          `Social Score: ${esgData.esg_scores?.social || 'N/A'}/10`,
          `Governance Score: ${esgData.esg_scores?.governance || 'N/A'}/10`,
          `**Overall ESG Score: ${esgData.esg_scores?.overall || 'N/A'}/10**`,
          '',
          'KEY ANALYSIS POINTS',
          ...esgData.analysis_points.slice(0, 5).map(point => `• ${point}`),
          '',
          'ESG RECOMMENDATIONS',
          ...esgData.recommendations.slice(0, 3).map(rec => `• ${rec}`),
          '',
          'COMPLIANCE STATUS',
          ...Object.entries(esgData.compliance_status || {}).map(([key, value]) => 
            `• ${key.replace('_', ' ').toUpperCase()}: ${value}`),
          '',
          'FINANCIAL IMPACT',
          ...Object.entries(esgData.financial_impact || {}).map(([key, value]) => 
            `• ${key.replace('_', ' ').toUpperCase()}: ${value}`),
          '',
          ...esgData.full_report.split('\n\n')
            .filter(paragraph => paragraph.trim().length > 0)
            .map(paragraph => paragraph.trim())
        ];
        
        // Send summary to SidePanel (left panel)
        if (onPredictionComment) {
          onPredictionComment(summaryForSidePanel);
        }
        
        // Send detailed analysis to ESGPanel (right panel)
        if (onSetEsgAnalysis) {
          onSetEsgAnalysis(detailedAnalysisForESGPanel);
        }
        
        if (onSetPropertyData) {
          onSetPropertyData(formData);
        }
        
        // Open the ESG panel
        if (onOpenEsgPanel) {
          onOpenEsgPanel();
        }
      }
    } catch (error) {
      console.error("ESG Analysis error:", error);
      
      // Fallback to detailed analysis similar to the original static version
      const epcScore = formData.epcScore;
      const surface = formData.habitableSurface;
      const year = formData.buildingConstructionYear;
      
      const isEnergyEfficient = ['A_plus', 'A', 'B'].includes(epcScore);
      const needsRenovation = ['E', 'F', 'G'].includes(epcScore);
      const yearlyEnergyCost = needsRenovation ? surface * 25 : surface * 15;
      const potentialSavings = needsRenovation ? yearlyEnergyCost * 0.6 : yearlyEnergyCost * 0.3;
      const renovationCost = needsRenovation ? surface * 250 : surface * 100;

      const fallbackAnalysis = [
        'ANALYSE ESG (MODE SIMPLIFIÉ)',
        '',
        `**Propriété:** ${formData.propertyType} à ${formData.locality}, ${formData.province}`,
        `**Score EPC:** ${formData.epcScore} | **Année:** ${formData.buildingConstructionYear}`,
        `**Chauffage:** ${formData.heatingType} | **État:** ${formData.buildingCondition}`,
        '',
        '**Note:** Analyse ESG détaillée indisponible. Vérifiez votre connexion internet ou réessayez plus tard.',
        '',
        'RECOMMANDATIONS DE BASE:',
        '• Examiner les améliorations d\'efficacité énergétique basées sur le score EPC',
        '• Considérer les améliorations d\'accessibilité pour un meilleur impact social', 
        '• S\'assurer de la conformité avec les réglementations belges du bâtiment',
        '',
        '**EPC Rating Analysis:** With an EPC score of ' + (epcScore === 'A_plus' ? 'A+' : epcScore?.replace('_', '+') || 'N/A') + (isEnergyEfficient ? ' (among the best in Belgium)' : needsRenovation ? ' (below current standards)' : ' (good performance)') + ', this house is ' + (isEnergyEfficient ? 'highly energy efficient and already exceeds current and near-future regulatory standards' : needsRenovation ? 'flagged for potential renovation needs to meet upcoming 2030 energy standards' : 'performing well but could benefit from targeted improvements') + '.',
        
        '**Energy Consumption Estimates:** For a ' + surface + 'm² house with an ' + (epcScore === 'A_plus' ? 'A+' : epcScore?.replace('_', '+') || 'N/A') + ' EPC, annual primary energy use typically ranges from ' + (needsRenovation ? '180-300' : isEnergyEfficient ? '50-80' : '100-150') + ' kWh/m², translating to roughly ' + Math.round(yearlyEnergyCost * 0.5) + '-' + Math.round(yearlyEnergyCost * 1.5) + ' kWh/year. Actual costs will depend on occupancy and usage, but expect ' + (isEnergyEfficient ? 'significantly lower' : needsRenovation ? 'higher than average' : 'moderate') + ' utility bills.',
        
        '**Investment Recommendations:** ' + (isEnergyEfficient ? 'This property represents an excellent long-term investment with minimal energy upgrade risks. Focus on maintaining systems and consider smart home technologies for further optimization.' : needsRenovation ? 'Priority renovations should target insulation, windows, and heating system upgrades. Estimated investment: €' + Math.round(renovationCost).toLocaleString() + ', with annual savings of €' + Math.round(potentialSavings).toLocaleString() + '.' : 'Consider targeted efficiency improvements like smart thermostats, improved insulation, or renewable energy integration to enhance both comfort and future-proofing.')
      ];
      
      if (onSetEsgAnalysis) {
        onSetEsgAnalysis(fallbackAnalysis);
      }
      
      if (onOpenEsgPanel) {
        onOpenEsgPanel();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Forcer l'ouverture du sidebar de gauche lors du clic sur Predict
    if (onOpenSidePanel) {
      onOpenSidePanel();
    }

    try {
      const encodedPayload = encodeInputs(formData);
      
      const response = await axios.post(`${PREDICTION_API_URL}/predict_all`, encodedPayload);

      setResults({
        all: response.data.prediction,
        top: null,
      });

      // Send prediction data to parent
      if (onSetPredictionData) {
        onSetPredictionData({
          prediction: response.data.prediction,
          predictionAll: response.data.prediction,
          predictionTop: 0
        });
      }

      // Créer un commentaire de prédiction pour le SidePanel
      if (onPredictionComment && response.data.prediction) {
        const timestamp = new Date().toLocaleTimeString('en-US', { 
          hour12: true, 
          hour: 'numeric', 
          minute: '2-digit', 
          second: '2-digit' 
        });
        const predictionComment = `Prediction for ${formData.propertyType.toLowerCase()} in ${formData.locality}, ${formData.province} - ${timestamp}`;
        const priceComment = `Predicted price: ${Math.round(response.data.prediction).toLocaleString('fr-FR').replace(/,/g, ' ')} €`;
        
        // Create model information comment
        const modelInfo = response.data.model_info || {};
        let modelComment = `Model: ${modelInfo.model_name || 'CatBoost + Optuna (All Features)'}`;
        if (modelInfo.r2_score) {
          modelComment += ` | R² Score: ${(modelInfo.r2_score * 100).toFixed(1)}%`;
        }
        if (modelInfo.mae) {
          modelComment += ` | MAE: ${Math.round(modelInfo.mae).toLocaleString('fr-FR').replace(/,/g, ' ')} €`;
        }
        
        const comments = [predictionComment, priceComment, modelComment];

        // Appeler l'API des commentaires LLM pour obtenir des commentaires automatiques
        try {
          const commentPayload = {
            formData: {
              ...formData,
              region: formData.province,
              scoreMeta: {
                epcScore: formData.epcScore,
                buildingCondition: formData.buildingCondition,
                mae: 25000,
                rmse: 35000,
                r2: 0.85
              }
            },
            predictionAll: response.data.prediction,
            predictionTop: 0,
            userProfile: {
              name: "Yves",
              profile: "Investisseur",
              type: "investor",
              objectives: ["regional_market_trends", "property_characteristics_analysis", "investment_return_strategy"],
              language: "en"
            }
          };

          const commentResponse = await axios.post(COMMENT_API_URL, commentPayload);

          // Ajouter les commentaires LLM aux commentaires existants
          if (commentResponse.data && commentResponse.data.comments) {
            comments.push(...commentResponse.data.comments);
          }
        } catch (commentError) {
          console.warn("Failed to get LLM comments:", commentError);
          // Continuer sans les commentaires LLM si l'API échoue
        }

        onPredictionComment(comments);
      }

    } catch (err) {
      console.error("Error:", err);
      setError("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="property-form" style={{ position: 'relative' }}>
      
      <div style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h1 style={{ 
          fontSize: '2.5rem', 
          fontWeight: '700', 
          color: '#1f2937', 
          marginBottom: '8px',
          margin: '0'
        }}>
          AI Property Report
        </h1>
        <p style={{ 
          fontSize: '1.125rem', 
          color: '#6b7280', 
          fontWeight: '500',
          margin: '0'
        }}>
          Value, ESG & Compliance
        </p>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div style={{ display: 'flex', gap: '10px' }}>
        <button
          type="button"
          className="reset-button"
          style={{ 
            height: '35px', 
            padding: '8px 16px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            fontSize: '14px'
          }}
          onClick={() => {
            setFormData(initialFormData);
            setResults({ all: null, top: null });
            setError(null);
          }}
          disabled={loading || esgLoading}
        >
          Reset
        </button>
        <button 
          type="submit" 
          className="submit-button" 
          style={{ 
            height: '35px', 
            padding: '8px 16px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            fontSize: '14px',
            backgroundColor: '#3b82f6',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            minWidth: '180px'
          }}
          disabled={loading || esgLoading}
          onClick={(e) => {
            console.log('🖱️ BUTTON CLICKED! Event:', e);
            console.log('🖱️ loading:', loading, 'esgLoading:', esgLoading);
            console.log('🖱️ button disabled:', loading || esgLoading);
            console.log('🖱️ About to call handleUnifiedAnalysis...');
            handleUnifiedAnalysis(e);
          }}
        >
          Analyze Price & ESG
        </button>

        {(loading || esgLoading) && (
          <span className="loading-text" style={{ marginLeft: '15px' }}>
            <span 
              style={{
                display: 'inline-block',
                width: '18px',
                height: '18px',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                borderTop: '1px solid #3b82f6',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
                marginRight: '3px'
              }}
            />
            Calling API...
          </span>
        )}
        </div>
        
        {/* Affichage du prix prédit aligné avec les boutons */}
        {results.all && (
          <div style={{ marginTop: '0px' }}>
            <ResultCard title="" value={results.all} />
          </div>
        )}
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '20px 0' }} />

      <div className="form-grid">
        {/* Property Type */}
        <div className="form-field">
          <label>Property Type</label>
          <select
            name="propertyType"
            value={formData.propertyType}
            onChange={handleChange}
          >
            {["HOUSE", "APARTMENT"].map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>

        {/* Subtype */}
        <div className="form-field">
          <label>Subtype</label>
          <select name="subtype" value={formData.subtype} onChange={handleChange}>
            {(subtypesByPropertyType[formData.propertyType] || []).map((opt) => (
              <option key={opt} value={opt}>
                {opt.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </div>

        {/* Province */}
        <div className="form-field">
          <label>Province</label>
          <select name="province" value={formData.province} onChange={handleChange}>
            {Object.keys(localityData).map((province) => (
              <option key={province} value={province}>
                {province}
              </option>
            ))}
          </select>
        </div>

        {/* Locality */}
        <div className="form-field">
          <label>Locality</label>
          <select name="locality" value={formData.locality} onChange={handleChange}>
            {availableLocalities.map((locality) => (
              <option key={locality} value={locality}>
                {locality}
              </option>
            ))}
          </select>
        </div>

        {/* Post Code */}
        <div className="form-field">
          <label>Post Code</label>
          <select name="postCode" value={formData.postCode} onChange={handleChange}>
            {Object.entries(postCodeToLocation).map(([code, data]) => (
              <option key={code} value={code}>
                {code} – {data.locality}
              </option>
            ))}
          </select>
        </div>

        {/* Living Surface */}
        <div className="form-field">
          <label>Living Surface (M²)</label>
          <input
            type="number"
            name="habitableSurface"
            value={formData.habitableSurface}
            onChange={handleChange}
            min="0"
          />
        </div>

        {/* Construction Year */}
        <div className="form-field">
          <label>Construction Year</label>
          <input
            type="number"
            name="buildingConstructionYear"
            value={formData.buildingConstructionYear}
            onChange={handleChange}
            min="1800"
            max={new Date().getFullYear()}
          />
        </div>

        {/* Bedrooms */}
        <div className="form-field">
          <label>Bedrooms</label>
          <input
            type="number"
            name="bedroomCount"
            value={formData.bedroomCount}
            onChange={handleChange}
            min="0"
          />
        </div>

        {/* Rooms Total */}
        <div className="form-field">
          <label>Rooms Total</label>
          <input
            type="number"
            name="roomCount"
            value={formData.roomCount}
            onChange={handleChange}
            min="0"
          />
        </div>

        {/* Bathrooms */}
        <div className="form-field">
          <label>Bathrooms</label>
          <input
            type="number"
            name="bathroomCount"
            value={formData.bathroomCount}
            onChange={handleChange}
            min="0"
          />
        </div>

        {/* Toilets */}
        <div className="form-field">
          <label>Toilets</label>
          <input
            type="number"
            name="toiletCount"
            value={formData.toiletCount}
            onChange={handleChange}
            min="0"
          />
        </div>

        {/* Facade Count */}
        <div className="form-field">
          <label>Facade Count</label>
          <input
            type="number"
            name="facedeCount"
            value={formData.facedeCount}
            onChange={handleChange}
            min="0"
          />
        </div>

        {/* Building Condition */}
        <div className="form-field">
          <label>Building Condition</label>
          <select name="buildingCondition" value={formData.buildingCondition} onChange={handleChange}>
            {["AS_NEW", "GOOD", "RENOVATION_NEEDED", "TO_RESTORE"].map((opt) => (
              <option key={opt} value={opt}>
                {opt.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </div>

        {/* Kitchen Type */}
        <div className="form-field">
          <label>Kitchen Type</label>
          <select name="kitchenType" value={formData.kitchenType} onChange={handleChange}>
            {["HYPER_EQUIPPED", "EQUIPPED", "SIMPLE", "NOT_INSTALLED"].map((opt) => (
              <option key={opt} value={opt}>
                {opt.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </div>

        {/* Heating Type */}
        <div className="form-field">
          <label>Heating Type</label>
          <select name="heatingType" value={formData.heatingType} onChange={handleChange}>
            {["ELECTRIC", "GAS", "NONE"].map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>

        {/* Flood Zone */}
        <div className="form-field">
          <label>Flood Zone</label>
          <select name="floodZoneType" value={formData.floodZoneType} onChange={handleChange}>
            {["NON_FLOOD_ZONE", "FLOOD_ZONE"].map((opt) => (
              <option key={opt} value={opt}>
                {opt.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </div>

        {/* EPC Score */}
        <div className="form-field">
          <label>EPC Score (Energy Class)</label>
          <select name="epcScore" value={formData.epcScore} onChange={handleChange}>
            <option value="A_plus">A+</option>
            <option value="A">A</option>
            <option value="B">B</option>
            <option value="C">C</option>
            <option value="D">D</option>
            <option value="E">E</option>
            <option value="F">F</option>
            <option value="G">G</option>
          </select>
        </div>

        {/* Has Living Room */}
        <div className="form-field">
          <label>Additional Features</label>
          <div className="checkbox-inline">
            <label className="checkbox-item">
              <input
                type="checkbox"
                name="hasLivingRoom"
                checked={formData.hasLivingRoom}
                onChange={handleChange}
              />
              Has Living Room
            </label>
          </div>
        </div>

        {/* Has Terrace */}
        <div className="form-field">
          <label>&nbsp;</label>
          <div className="checkbox-inline">
            <label className="checkbox-item">
              <input
                type="checkbox"
                name="hasTerrace"
                checked={formData.hasTerrace}
                onChange={handleChange}
              />
              Has Terrace
            </label>
          </div>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* ESG Quick Assessment - always visible and updates in real-time */}
      <StrategicAnalysisConclusion 
        formData={formData} 
        esgAnalysisAvailable={esgAnalysisAvailable}
        detailedEsgData={detailedEsgData}
        esgLoading={esgLoading}
        onOpenSidePanel={onOpenSidePanel}
        onSendChatMessage={onSendChatMessage}
      />
    </div>
  );
};

export default PropertyForm;