/**
 * Belgian Real Estate ESG Agent
 * Specialized AI assistant for sustainability and regulatory compliance
 */

import React, { useState, useEffect } from 'react';

const BelgianESGAgent = ({ propertyData, estimatedPrice, onAnalysisComplete }) => {
  const [messages, setMessages] = useState([
    {
      type: 'agent',
      content: propertyData ? 
        `Property Analysis Ready\n\nProperty: ${propertyData.habitableSurface}m² ${propertyData.propertyType} in ${propertyData.locality}\nEstimated Value: €${estimatedPrice?.toLocaleString()}\nEPC Score: ${propertyData.epcScore}\n\nGenerating detailed ESG analysis...` :
        "Hello! I'm your Belgian real estate ESG advisor. I can help you with:\n\nEnergy Performance (EPC)\nGrants and subsidies\nValue impact\nSustainable renovations\nRegulatory compliance\n\nWhat's your question?"
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useRealAPI, setUseRealAPI] = useState(false); // Toggle between simulation and real API

  // Backend API configuration
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8010';

  // Call real Azure OpenAI API
  const callRealAPI = async (userMessage) => {
    try {
      const response = await fetch(`${API_BASE_URL}/esg_agent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [
            {
              role: 'user',
              content: userMessage
            }
          ]
        })
      });

      if (!response.ok) {
        throw new Error(`API call failed: ${response.status}`);
      }

      const data = await response.json();
      return data.response;
    } catch (error) {
      console.error('API Error:', error);
      return `I apologize, but I'm currently unable to connect to the advanced AI system. However, I can still provide some general ESG guidance based on Belgian real estate regulations.

For immediate assistance, please try one of the demo scenarios above, or ask about:
- EPC energy classes and their impact
- 2030 rental restrictions  
- Available renovation grants
- Sustainable investment strategies

Error details: ${error.message}`;
    }
  };

  // Auto-generate ESG analysis when property data is available
  useEffect(() => {
    if (propertyData && estimatedPrice) {
      setTimeout(() => {
        const analysis = generateDetailedESGAnalysis(propertyData, estimatedPrice);
        setMessages(prev => [...prev, {
          type: 'agent',
          content: analysis
        }]);
        if (onAnalysisComplete) {
          onAnalysisComplete(analysis);
        }
      }, 1500); // Simulate processing time
    }
  }, [propertyData, estimatedPrice, onAnalysisComplete]);

  // Predefined scenarios for demo
  const demoScenarios = [
    {
      label: "1960 House Class F",
      query: "I have a 1960 house rated F in Brussels. What's the price impact and what should I do?"
    },
    {
      label: "Insulation Grants",
      query: "What grants can I get for insulating a house in Wallonia?"
    },
    {
      label: "Heat Pump ROI",
      query: "ROI of a heat pump vs gas boiler for 100m² apartment"
    },
    {
      label: "2030 Deadlines",
      query: "My rental property is class G, what happens in 2030?"
    }
  ];

  const handleSendMessage = async (message = inputValue) => {
    if (!message.trim()) return;

    const userMessage = { type: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      let response;
      if (useRealAPI) {
        // Use real Azure OpenAI API
        response = await callRealAPI(message);
      } else {
        // Use simulated responses
        response = generateESGResponse(message);
      }
      
      setMessages(prev => [...prev, { type: 'agent', content: response }]);
    } catch (error) {
      console.error('Error generating response:', error);
      setMessages(prev => [...prev, { 
        type: 'agent', 
        content: 'Sorry, I encountered an error. Please try again or contact support.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const generateESGResponse = (query) => {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('class f') || lowerQuery.includes('1960')) {
      return `1960 House Class F Analysis - Brussels

Current Price Impact
- Depreciation: -15 to -20% vs class C
- Market price reduced by ~30-40k€

Regulatory Risks
- 2030: Ban on Class F rentals
- Progressive loss of rental value
- Resale difficulties without renovation

Solutions & Grants (Brussels)
- Roof insulation: Grant up to 15€/m²
- Wall insulation: Grant up to 40€/m²
- Heat pump: Grant 2000-4000€
- Total available grants: ~15-25k€

Estimated ROI
- Renovation cost: 45-60k€
- Grants: -20k€
- Net cost: 25-40k€
- Value gain: +50-70k€
- Net ROI: +25-30k€

Recommendation
Plan renovation before 2028 to optimize grants and value.`;
    }
    
    if (lowerQuery.includes('grant') || lowerQuery.includes('subsidy') || lowerQuery.includes('wallonia')) {
      return `💰 **Wallonia Insulation Grants 2025**

🏠 **Housing Grants**
- Roof insulation: 15-30€/m² (max 3000€)
- Wall insulation: 25-50€/m² (max 5000€) 
- Floor insulation: 10-20€/m² (max 2000€)
- High-performance windows: 15-30€/m² (max 2500€)

🔥 **Heating Grants**
- Air/water heat pump: 2000-4000€
- Geothermal heat pump: 4000-6000€
- Biomass boiler: 1500-3000€
- Solar water heater: 1000-2000€

📋 **Eligibility Conditions**
- Income < thresholds (varies by municipality)
- Mandatory energy audit
- Certified contractors only
- Performance gain ≥ 1 EPC class

💡 **Cumulative Possibilities**
- Regional + municipal grants
- 6% VAT reduction (renovation)
- 30% tax deduction (max 3830€/year)

Total cumulative possible: **15-30k€** depending on project!`;
    }
    
    if (lowerQuery.includes('pump') || lowerQuery.includes('roi') || lowerQuery.includes('heat')) {
      return `🔥 **Heat Pump vs Gas ROI (100m²)**

💰 **Initial Investment**
- Air/water heat pump: 12-18k€
- Installation + adaptation: 3-5k€
- **Total: 15-23k€**
- Grants deducted: -3k€
- **Net cost: 12-20k€**

📊 **Annual Costs (100m²)**
- Current gas: ~1200-1500€/year
- Heat pump: ~800-1000€/year
- **Savings: 400-500€/year**

⚡ **Financial ROI**
- 20-year savings: 8-10k€
- Property value gain: +10-15k€
- **Total ROI: 18-25k€**
- **Payback period: 8-12 years**

🌱 **Environmental Impact**
- CO₂ reduction: -2.5 tons/year
- Transition class E → B/A
- 2030+ compliance guaranteed

📈 **Energy Price Evolution**
- Gas: +3-5%/year projected
- Renewable electricity: stability
- **Improved ROI over time**

🎯 **Verdict: Profitable investment** 
Especially with grants and 2030 perspective!`;
    }
    
    if (lowerQuery.includes('2030') || lowerQuery.includes('deadline') || lowerQuery.includes('rental')) {
      return `2030 Deadlines: Class G Rental Properties

Regulatory Timeline
- 2026: Mandatory energy audit for rentals
- 2028: Ban on new Class G leases
- 2030: Total ban on F & G rentals
- 2035: Probable extension to Class E

Immediate Consequences
- Inability to rent legally
- Fines: 500-2000€ penalty
- Total loss of rental income
- Property value depreciation (-30-40%)

Critical Financial Impact
- 200k€ property → Residual value 120-140k€
- Net loss: 60-80k€
- Avoidable renovation cost: 40-50k€

Urgent Action Plan
1. EPC Audit before end 2025
2. Renovation planning 2026-2027
3. Grant optimization (end 2027)
4. Complete renovation before 2028

Time Remaining: 3-4 years
- The longer you wait, the more it costs
- Reduced grants after 2027
- Saturated contractors near 2030

Recommended Action
IMMEDIATE: Plan now or sell before depreciation!`;
    }

    // Default response
    return `ESG Analysis in progress...

Thank you for your question! As a Belgian real estate ESG specialist, I can help you with:

Energy Performance
- EPC audit and certification
- Energy class improvement
- Sale/rental price impact

Financial Optimization
- Available grants and subsidies
- Sustainable renovation ROI
- Tax planning

Regulatory Compliance
- 2030-2035 deadlines
- Owner obligations
- Compliance strategies

Sustainability
- Eco-friendly solutions
- CO₂ reductions
- Innovative technologies

Can you specify your situation (property type, location, EPC class) for a personalized analysis?`;
  };

  const generateDetailedESGAnalysis = (propertyData, estimatedPrice) => {
    const epcScore = propertyData.epcScore;
    const surface = propertyData.habitableSurface;
    const year = propertyData.buildingConstructionYear;
    const locality = propertyData.locality;
    const province = propertyData.province;
    
    // Calculate energy efficiency metrics
    const isOldBuilding = year < 1980;
    const isEnergyEfficient = ['A_plus', 'A', 'B'].includes(epcScore);
    const needsRenovation = ['E', 'F', 'G'].includes(epcScore);
    
    // Calculate potential savings and renovations
    const yearlyEnergyCost = needsRenovation ? surface * 25 : surface * 15;
    const potentialSavings = needsRenovation ? yearlyEnergyCost * 0.6 : yearlyEnergyCost * 0.3;
    const renovationCost = needsRenovation ? surface * 250 : surface * 100;

    return `Detailed ESG Analysis - ${locality}, ${province}

Property Overview
• ${surface}m² ${propertyData.propertyType.toLowerCase()} built in ${year}
• Current EPC: ${epcScore.replace('_', '+')}
• Estimated Value: €${estimatedPrice.toLocaleString()}

Energy Performance
${isEnergyEfficient ? 
  `Excellent Performance!
• Low energy costs (~€${Math.round(yearlyEnergyCost)}/year)
• High market value retention
• Compliant with 2030+ regulations` :
  needsRenovation ?
  `Renovation Needed
• High energy costs (~€${Math.round(yearlyEnergyCost)}/year)
• Potential savings: €${Math.round(potentialSavings)}/year
• Regulatory risk for rentals post-2030` :
  `Good Performance
• Moderate energy costs (~€${Math.round(yearlyEnergyCost)}/year)
• Room for improvement: €${Math.round(potentialSavings)}/year savings`
}

Financial Impact
• Current impact: ${needsRenovation ? '-15% to -20%' : isEnergyEfficient ? '+5% to +10%' : 'neutral to +5%'}
• Post-renovation value: +€${Math.round(renovationCost * 0.8).toLocaleString()}
• ROI timeline: ${needsRenovation ? '7-10 years' : '10-15 years'}

Renovation Recommendations
${needsRenovation ? 
  `Priority investments:
• Insulation (roof/walls): €${Math.round(surface * 80)}-${Math.round(surface * 120)}
• High-efficiency heating: €${Math.round(surface * 60)}-${Math.round(surface * 100)}
• Windows replacement: €${Math.round(surface * 40)}-${Math.round(surface * 80)}` :
  `Optimization opportunities:
• Smart heating control: €2,000-5,000
• Solar panels: €8,000-15,000
• Ventilation upgrade: €3,000-7,000`
}

Belgian Grants Available
• ${province} region: Up to €4,000 base grant
• Federal tax deduction: 30% on energy works
• Municipality bonus: €500-2,000 additional

Market Outlook
• Energy-efficient homes: +5-8% demand growth
• ESG compliance: Critical for rental market
• Carbon footprint: ${isEnergyEfficient ? 'Low' : needsRenovation ? 'High - action needed' : 'Moderate'}

Next Steps: ${needsRenovation ? 'Schedule energy audit → Apply for grants → Execute renovations' : 'Consider optimization upgrades for added value'}`;
  };

  return (
    <div className="max-w-6xl mx-auto bg-white rounded-xl shadow-xl">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-6 rounded-t-xl">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-2xl font-bold">
              ESG Agent - Sustainable Real Estate Advisor
            </h2>
            <p className="text-green-100 text-sm mt-2">
              Specialized in Belgian regulations, grants and energy performance
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-green-100">Demo Mode</span>
              <button
                onClick={() => setUseRealAPI(!useRealAPI)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  useRealAPI ? 'bg-green-400' : 'bg-gray-300'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    useRealAPI ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className="text-sm text-green-100">AI Mode</span>
            </div>
          </div>
        </div>
      </div>

      {/* Demo Scenarios */}
      <div className="p-6 bg-gray-50 border-b">
        <p className="text-sm text-gray-700 mb-3 font-medium">Try these demo scenarios:</p>
        <div className="flex flex-wrap gap-3">
          {demoScenarios.map((scenario, index) => (
            <button
              key={index}
              onClick={() => handleSendMessage(scenario.query)}
              className="bg-white border border-gray-300 text-gray-700 px-4 py-2 rounded-lg text-sm hover:bg-blue-50 hover:border-blue-300 transition-all duration-200 shadow-sm hover:shadow-md"
              title={scenario.query}
            >
              {scenario.label}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="h-[600px] overflow-y-auto p-6 space-y-6 bg-gray-50">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-4xl px-6 py-4 rounded-xl shadow-sm ${
                message.type === 'user'
                  ? 'bg-blue-600 text-white ml-12'
                  : 'bg-white text-gray-800 mr-12 border border-gray-200'
              }`}
            >
              <div className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white text-gray-800 px-6 py-4 rounded-xl shadow-sm border border-gray-200 mr-12">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                <span className="text-sm text-gray-600">ESG Agent is analyzing...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-6 bg-white border-t border-gray-200 rounded-b-xl">
        <div className="flex space-x-4">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Ask about ESG compliance, energy ratings, grants, or renovations..."
            className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            disabled={isLoading}
          />
          <button
            onClick={() => handleSendMessage()}
            disabled={isLoading || !inputValue.trim()}
            className="bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
          >
            Send
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          {useRealAPI ? 
            'AI Mode: Connected to Azure OpenAI for advanced responses. Ensure backend API is running.' :
            'Demo Mode: Using pre-configured Belgian ESG scenarios and regulations.'
          }
        </p>
      </div>
    </div>
  );
};

export default BelgianESGAgent;
