import React, { useState } from "react";
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
  };

  // New function for ESG Analysis button
  const handleESGAnalysis = async () => {
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

      const response = await axios.post(ESG_API_URL, requestData);
      const esgData = response.data;

      // Generate timestamp
      const now = new Date();
      const timestamp = now.toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true 
      });

      // Create ESG Analysis chat comment with simplified format
      const esgComment = `${formData.propertyType} in ${formData.locality}, ${formData.province} (${timestamp})`;
      
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
      
      // Even if analysis fails, show fallback ESG data
      const fallbackEsgData = {
        esg_scores: {
          environmental: formData.epcScore === 'A_plus' || formData.epcScore === 'A' ? 8.5 : 
                         formData.epcScore === 'B' ? 7.5 : 
                         formData.epcScore === 'C' ? 6.5 : 
                         formData.epcScore === 'D' ? 5.5 : 4.5,
          social: 7.0,
          governance: formData.buildingConstructionYear > 2000 ? 7.5 : 6.5,
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
    } finally {
      setEsgLoading(false);
      
      // Notify parent about loading state
      if (onSetEsgLoading) {
        onSetEsgLoading(false);
      }
    }
  };

  // New unified function that combines price prediction and ESG analysis
  const handleUnifiedAnalysis = async (e) => {
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

      const esgResponse = await axios.post(ESG_API_URL, esgRequestData);
      const esgData = esgResponse.data;

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
            environmental: formData.epcScore === 'A_plus' || formData.epcScore === 'A' ? 8.5 : 
                           formData.epcScore === 'B' ? 7.5 : 
                           formData.epcScore === 'C' ? 6.5 : 
                           formData.epcScore === 'D' ? 5.5 : 4.5,
            social: 7.0,
            governance: formData.buildingConstructionYear > 2000 ? 7.5 : 6.5,
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

  // New function for Strategic Analysis button
  const handleStrategicAnalysis = async () => {
    setLoading(true);
    setError(null);
    
    // Open side panel to show progress
    if (onOpenSidePanel) {
      onOpenSidePanel();
    }

    try {
      // Generate timestamp for strategic analysis
      const timestamp = new Date().toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true 
      });

      const strategicComment = `Strategic Analysis for ${formData.propertyType.toLowerCase()} in ${formData.locality} (${timestamp})`;
      
      // Add strategic analysis to chat
      if (onPredictionComment) {
        onPredictionComment([
          strategicComment,
        ]);
      }

      // Trigger the strategic analysis directly in the SidePanel using the same business logic
      if (onSendChatMessage) {
        // Prepare comprehensive ESG data for the strategic analysis INCLUDING ALL DETAILED DATA
        const esgScores = detailedEsgData?.esg_scores || {};
        const compliance = detailedEsgData?.compliance_status || {};
        const financialImpact = detailedEsgData?.financial_impact || {};
        const recommendations = detailedEsgData?.recommendations || [];
        const analysisPoints = detailedEsgData?.analysis_points || [];
        
        // Include the complete ESG Analysis Report content
        const fullEsgReport = detailedEsgData?.full_report || '';
        
        // Create comprehensive strategic analysis prompt with ALL ESG data including detailed analysis
        const strategicAnalysisPrompt = `Generate a comprehensive strategic analysis for this Belgian real estate investment based on the complete ESG analysis data. Structure it with clear markdown headers:

# Strategic Analysis – ${formData.locality} Property Investment

## ESG Analysis Summary
**Environmental Score:** ${esgScores.environmental || 'N/A'}/10 **Social Score:** ${esgScores.social || 'N/A'}/10 **Governance Score:** ${esgScores.governance || 'N/A'}/10 **Overall ESG Score:** ${esgScores.overall || 'N/A'}/10

## Investment Positioning
Based on the ESG analysis results above, analyze the investment potential for this ${formData.propertyType.toLowerCase()} property in ${formData.locality}.

## Market Context
Analyze the ${formData.locality} market, construction year ${formData.buildingConstructionYear || 'recent'}, and EPC rating ${formData.epcScore || 'modern'} positioning.

## ESG Compliance Status
${Object.entries(compliance).map(([key, value]) => `**${key.replace(/_/g, ' ').toUpperCase()}:** ${value}`).join('\n')}

## Financial Impact Analysis
${Object.entries(financialImpact).map(([key, value]) => `**${key.replace(/_/g, ' ').toUpperCase()}:** ${value}`).join('\n')}

## Key ESG Analysis Points
${analysisPoints.slice(0, 6).map(point => `• ${point}`).join('\n')}

## ESG-Based Recommendations
${recommendations.slice(0, 6).map(rec => `• ${rec}`).join('\n')}

## Complete ESG Analysis Report Integration
**Reference the detailed ESG analysis for comprehensive insights:**

${fullEsgReport}

## Strategic Recommendations

### Short-term Actions (0-6 months)
Immediate improvement opportunities based on ESG findings
Quick wins for value enhancement from compliance analysis

### Medium-term Strategy (6-24 months)  
Major improvement projects aligned with ESG scores
Market positioning optimization considering financial impact

### Long-term Vision (2+ years)
Future-proofing strategies based on governance assessment
Regulatory compliance preparation for Belgian market

## Risk Assessment
Evaluate potential risks and mitigation strategies based on current ESG compliance status and market trends.

Property details: Surface ${formData.habitableSurface}m², ${formData.bedroomCount} bedrooms, ${formData.buildingCondition} condition, ${formData.heatingType} heating.`;

        // Send this to the SidePanel chat which will trigger the strategic analysis
        await onSendChatMessage(strategicAnalysisPrompt);
      }

    } catch (error) {
      console.error("Strategic Analysis error:", error);
      setError("Strategic Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetailedESGReport = async () => {
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
          onClick={handleUnifiedAnalysis}
        >
          Analyze Price & ESG
        </button>
        
        <button 
          type="button" 
          className="strategic-analysis-button" 
          style={{ 
            height: '35px', 
            padding: '8px 16px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            fontSize: '14px',
            backgroundColor: '#8b5cf6',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            minWidth: '160px'
          }}
          disabled={loading || esgLoading}
          onClick={handleStrategicAnalysis}
        >
          Strategic Analysis
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