/**
 * Belgian Real Estate ESG Agent
 * Specialized AI assistant for sustainability and regulatory compliance
 */

import React, { useState } from 'react';

const BelgianESGAgent = () => {
  const [messages, setMessages] = useState([
    {
      type: 'agent',
      content: "Hello! I'm your Belgian real estate ESG advisor. I can help you with:\n\nEnergy Performance (EPC)\nGrants and subsidies\nValue impact\nSustainable renovations\nRegulatory compliance\n\nWhat's your question?"
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

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

    // Simulate AI response with Belgian context
    setTimeout(() => {
      const response = generateESGResponse(message);
      setMessages(prev => [...prev, { type: 'agent', content: response }]);
      setIsLoading(false);
    }, 1500);
  };

  const generateESGResponse = (query) => {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('class f') || lowerQuery.includes('1960')) {
      return `**1960 House Class F Analysis - Brussels**

📉 **Current Price Impact**
- Depreciation: -15 to -20% vs class C
- Market price reduced by ~30-40k€

⚠️ **Regulatory Risks**
- 2030: Ban on Class F rentals
- Progressive loss of rental value
- Resale difficulties without renovation

💰 **Solutions & Grants (Brussels)**
- Roof insulation: Grant up to 15€/m²
- Wall insulation: Grant up to 40€/m²
- Heat pump: Grant 2000-4000€
- Total available grants: ~15-25k€

📈 **Estimated ROI**
- Renovation cost: 45-60k€
- Grants: -20k€
- Net cost: 25-40k€
- Value gain: +50-70k€
- **Net ROI: +25-30k€**

🎯 **Recommendation**
Plan renovation before 2028 to optimize grants and value.`;
    }
    
    if (lowerQuery.includes('grant') || lowerQuery.includes('subsidy') || lowerQuery.includes('wallonia')) {
      return `💰 **Wallonia Insulation Grants 2025**

**Housing Grants**
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
      return `⚠️ **2030 Deadlines: Class G Rental Properties**

📅 **Regulatory Timeline**
- **2026**: Mandatory energy audit for rentals
- **2028**: Ban on new Class G leases
- **2030**: Total ban on F & G rentals
- **2035**: Probable extension to Class E

🚫 **Immediate Consequences**
- Inability to rent legally
- Fines: 500-2000€ penalty
- Total loss of rental income
- Property value depreciation (-30-40%)

💸 **Critical Financial Impact**
- 200k€ property → Residual value 120-140k€
- Net loss: **60-80k€**
- Avoidable renovation cost: 40-50k€

🚀 **Urgent Action Plan**
1. **EPC Audit** before end 2025
2. **Renovation planning** 2026-2027
3. **Grant optimization** (end 2027)
4. **Complete renovation** before 2028

⏰ **Time Remaining: 3-4 years**
- The longer you wait, the more it costs
- Reduced grants after 2027
- Saturated contractors near 2030

🎯 **Recommended Action**
**IMMEDIATE**: Plan now or sell before depreciation!`;
    }

    // Default response
    return `🤖 **ESG Analysis in progress...**

Thank you for your question! As a Belgian real estate ESG specialist, I can help you with:

**Energy Performance**
- EPC audit and certification
- Energy class improvement
- Sale/rental price impact

💰 **Financial Optimization**
- Available grants and subsidies
- Sustainable renovation ROI
- Tax planning

⚖️ **Regulatory Compliance**
- 2030-2035 deadlines
- Owner obligations
- Compliance strategies

🌱 **Sustainability**
- Eco-friendly solutions
- CO₂ reductions
- Innovative technologies

Can you specify your situation (property type, location, EPC class) for a personalized analysis?`;
  };

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-4 rounded-t-lg">
        <h2 className="text-xl font-semibold flex items-center space-x-2">
          <span>🌱</span>
          <span>ESG Agent - Sustainable Real Estate Advisor</span>
        </h2>
        <p className="text-green-100 text-sm mt-1">
          Specialized in Belgian regulations, grants and energy performance
        </p>
      </div>

      {/* Demo Scenarios */}
      <div className="p-4 bg-gray-50 border-b">
        <p className="text-sm text-gray-600 mb-2">💡 Try these demo scenarios:</p>
        <div className="flex flex-wrap gap-2">
          {demoScenarios.map((scenario, index) => (
            <button
              key={index}
              onClick={() => handleSendMessage(scenario.query)}
              className="bg-white border border-gray-300 text-gray-700 px-3 py-1 rounded-full text-xs hover:bg-blue-50 hover:border-blue-300 transition-colors"
              title={scenario.query}
            >
              {scenario.label}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="h-96 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.type === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <div className="whitespace-pre-wrap text-sm">{message.content}</div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 text-gray-800 px-4 py-2 rounded-lg">
              <div className="flex items-center space-x-1">
                <div className="animate-pulse">🤖</div>
                <span className="text-sm">Analysis in progress...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Ask your real estate ESG question..."
            className="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
            disabled={isLoading}
          />
          <button
            onClick={() => handleSendMessage()}
            disabled={isLoading || !inputValue.trim()}
            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
          >
            🚀
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          💡 Specialized advice on EPC regulations, Belgian grants and sustainable strategies
        </p>
      </div>
    </div>
  );
};

export default BelgianESGAgent;
