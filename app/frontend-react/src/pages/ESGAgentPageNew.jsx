import React, { useState, useEffect, useRef } from 'react';
import { useUser } from '../contexts/UserContext';

const ESGAgentPageNew = () => {
  const { userProfile, user } = useUser();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useRealAPI, setUseRealAPI] = useState(false);
  const messagesEndRef = useRef(null);

  // Backend API configuration - Use EXISTING API
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8010';

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize welcome message
  useEffect(() => {
    if (userProfile) {
      setMessages([{
        type: 'agent',
        content: `Welcome ${user?.name || 'back'} to your Belgian Real Estate ESG Advisor!

I'm here to provide personalized advice for ${userProfile.user_role}s on:
- Energy Performance Certificates (EPC) and regulations
- Belgian grants and subsidies for renovations  
- 2030-2035 compliance deadlines
- Sustainable investment strategies
- Property value optimization

Click any question from the sidebar to get started, or ask me anything about ESG compliance for Belgian real estate!`
      }]);
    }
  }, [userProfile, user]);

  // Call real Azure OpenAI API
  const callRealAPI = async (userMessage) => {
    try {
      console.log(`Calling ESG Agent API: ${API_BASE_URL}/esg_agent`);
      console.log('Request payload:', userMessage);
      
      const response = await fetch(`${API_BASE_URL}/esg_agent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
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

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error Response:', response.status, errorText);
        throw new Error(`API request failed with status ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log('API Response data:', data);
      return data.response || data.message || data.content || data;
      
    } catch (error) {
      console.error('API Call Error:', error);
      throw error; // Re-throw to use fallback mode
    }
  };

  // Generate simulated ESG response
  const generateESGResponse = (query) => {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('class f') || lowerQuery.includes('epc') && lowerQuery.includes('risk')) {
      return `EPC Class F Investment Risk Analysis

Regulatory Timeline
- 2026: Mandatory energy audit for all rentals
- 2028: Ban on new Class F lease agreements
- 2030: Complete ban on F & G rental properties
- 2035: Probable extension to Class E

Financial Impact for ${userProfile?.user_role || 'Property Stakeholders'}
- Current market depreciation: -15% to -25% vs Class C
- Post-2030 rental ban: Total loss of rental income
- Estimated property value drop: -30% to -40%

${userProfile?.user_role === 'Investor' ? 
`Investment Strategy
- Avoid Class F properties unless renovation budget is secured
- Factor renovation costs (€250-400/m²) into purchase price
- Target acquisition 2-3 years before renovation to maximize grants` :
`Renovation Priority
- Begin planning immediately for 2026-2027 execution
- Secure grant applications before demand peak
- Budget €40-60k for typical 150m² property renovation`}

Available Solutions
- Insulation upgrades: €15-25k (grants up to 40%)
- Heat pump installation: €8-15k (grants up to €4k)
- Window replacement: €10-20k (grants up to 30%)

Recommendation: ${userProfile?.user_role === 'Investor' ? 'Avoid or price in full renovation costs' : 'Start renovation planning now to avoid 2030 penalties'}`;
    }

    if (lowerQuery.includes('grant') || lowerQuery.includes('subsid')) {
      return `Belgian Renovation Grants & Subsidies 2025

Federal Level
- Tax reduction: 30% on energy renovation costs
- Maximum deduction: €7,500 per year
- Valid for: insulation, heating systems, solar panels

Regional Grants (Flanders)
- Roof insulation: Up to €15/m² (max €3,000)
- Wall insulation: Up to €40/m² (max €4,000)  
- Heat pump: €2,000-€4,000 depending on type
- Solar panels: €150 per kWp installed

Regional Grants (Wallonia)
- Energy renovation: Up to €6,000 base grant
- Low-income bonus: Additional €3,000
- Heat pump: €3,000-€5,000 depending on efficiency

Regional Grants (Brussels)
- Renowatt program: Up to €5,000 for comprehensive renovation
- Insulation grants: €35/m² for walls, €15/m² for roof
- Heat pump: €4,000 maximum

${userProfile?.user_role === 'Investor' ? 
`ROI Optimization for Investors
- Combine multiple grants to maximize return
- Typical total grants available: €10-20k per property
- Renovation cost: €40-60k, Net cost after grants: €25-40k
- Property value increase: €50-80k, Net ROI: €15-25k` :
`Application Strategy
- Apply early - grants have annual budget limits
- Combine regional + federal incentives
- Use certified contractors to qualify for maximum grants`}

Next Steps
1. Verify your region's specific grant amounts
2. Get energy audit to determine eligible improvements  
3. Apply for grants before starting work
4. Keep all receipts for tax deduction claims`;
    }

    if (lowerQuery.includes('2030') || lowerQuery.includes('deadline') || lowerQuery.includes('compliance')) {
      return `2030 ESG Compliance Deadlines Belgium

Critical Dates
- 2026: Energy Performance Certificate mandatory for all rentals
- 2028: No new rental contracts for Class F & G properties
- 2030: Complete rental ban for Class F & G properties
- 2035: Expected expansion to include Class E properties

${userProfile?.user_role === 'Investor' ? 
`Investment Impact Analysis
- Rental yield protection: Upgrade before 2028 to maintain income
- Property values: Non-compliant properties losing 20-40% value
- Market opportunity: Compliant properties gaining rental premium
- Exit strategy: Sell non-compliant properties before 2028 value drop` :
`Compliance Action Plan
- 2025: Get EPC assessment and renovation quotes
- 2026: Secure grants and start major renovations  
- 2027: Complete renovations and obtain new EPC
- 2028: Property fully compliant for continued rental`}

Penalties for Non-Compliance
- Administrative fines: €500-€2,000 per violation
- Loss of rental income: 100% from 2030
- Insurance issues: Reduced coverage for non-compliant properties
- Resale difficulties: Limited buyer pool for non-compliant properties

Compliance Strategy
- Phase 1: Energy audit and improvement planning
- Phase 2: Grant applications and contractor selection
- Phase 3: Renovation execution with milestone monitoring
- Phase 4: New EPC certification and compliance verification

Time Remaining: ${Math.ceil((new Date('2030-01-01') - new Date()) / (1000 * 60 * 60 * 24 * 365))} years to ensure full compliance`;
    }

    // Default comprehensive response
    return `ESG Analysis for ${userProfile?.user_role || 'Property Stakeholders'}

Based on your profile as a ${userProfile?.user_role || 'property stakeholder'}, here's what you need to know:

Current Market Context
- Belgian real estate undergoing major ESG transformation
- 2030 regulations creating compliance urgency
- Grant availability at historic highs until 2027
- Property values increasingly tied to energy performance

${userProfile?.user_role === 'Investor' ? 
`Investment Recommendations
- Focus on Class A-C properties or budget full renovation costs
- Factor €250-400/m² renovation costs into acquisition analysis
- Target properties in grant-eligible regions for maximum ROI
- Consider portfolio diversification across energy classes` :
userProfile?.user_role === 'Real Estate Agent' ?
`Client Advisory Points
- Educate sellers on renovation value-add before listing
- Help buyers understand long-term compliance costs
- Position energy-efficient properties as premium offerings
- Develop expertise in grant application guidance` :
`Property Owner Action Items
- Schedule energy audit within 6 months
- Research available grants in your specific region
- Get renovation quotes from certified contractors
- Plan renovation timeline to avoid 2030 penalties`}

Key Focus Areas
1. Energy Performance Certificate (EPC) optimization
2. Regional grant and subsidy maximization
3. 2030 compliance planning and execution
4. Property value preservation and enhancement

Would you like me to dive deeper into any specific aspect of Belgian real estate ESG compliance?`;
  };

  // Handle sending messages
  const handleSendMessage = async (message = inputValue, fromSidebar = false) => {
    if (!message.trim()) return;

    const userMessage = { type: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      let response;
      if (useRealAPI) {
        try {
          response = await callRealAPI(message);
        } catch (apiError) {
          console.log('API failed, falling back to local simulation');
          response = generateESGResponse(message) + '\n\n*Note: Using local simulation as the backend API is not available.*';
        }
      } else {
        response = generateESGResponse(message);
      }
      
      setMessages(prev => [...prev, { type: 'agent', content: response }]);
      
      // Auto-scroll to response if message came from sidebar
      if (fromSidebar) {
        setTimeout(scrollToBottom, 100);
      }
    } catch (error) {
      console.error('Error generating response:', error);
      setMessages(prev => [...prev, { 
        type: 'agent', 
        content: 'Sorry, I encountered an error generating a response. Please try asking a different question or check one of the suggested questions from the sidebar.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle question tile click
  const handleQuestionClick = (question) => {
    handleSendMessage(question, true);
  };

  if (!userProfile) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your ESG advisor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Left Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        {/* Sidebar Header */}
        <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-6">
          <h2 className="text-xl font-bold mb-2">
            ESG Agent – Sustainable Real Estate Advisor
          </h2>
          <p className="text-green-100 text-sm">
            Personalized for {userProfile.user_role}s
          </p>
        </div>

        {/* User Info */}
        <div className="p-4 bg-blue-50 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-semibold">
              {user?.name?.charAt(0) || user?.email?.charAt(0) || 'U'}
            </div>
            <div>
              <p className="font-medium text-gray-800">{user?.name || 'User'}</p>
              <p className="text-sm text-gray-600">{userProfile.user_role}</p>
            </div>
          </div>
        </div>

        {/* API Mode Toggle */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-700">AI Mode</span>
            <button
              onClick={() => setUseRealAPI(!useRealAPI)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                useRealAPI ? 'bg-green-500' : 'bg-gray-300'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  useRealAPI ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            {useRealAPI ? 'Azure OpenAI enabled' : 'Demo responses'}
          </p>
        </div>

        {/* Question Tiles */}
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-4 uppercase tracking-wide">
            Suggested Questions
          </h3>
          <div className="space-y-3">
            {userProfile.suggested_questions?.map((question, index) => (
              <button
                key={index}
                onClick={() => handleQuestionClick(question)}
                className="w-full text-left p-4 bg-gray-50 hover:bg-blue-50 hover:border-blue-200 border border-gray-200 rounded-lg transition-all duration-200 text-sm"
              >
                <div className="font-medium text-gray-800 mb-1">
                  {question}
                </div>
                <div className="text-xs text-gray-500">
                  Click to ask this question
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <p className="text-xs text-gray-500 text-center">
            Specialized in Belgian regulations and ESG compliance
          </p>
        </div>
      </div>

      {/* Right Chat Section */}
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-800">
                ESG Real Estate Advisor
              </h1>
              <p className="text-sm text-gray-600">
                AI-powered insights for sustainable Belgian real estate
              </p>
            </div>
            <div className="text-sm text-gray-500">
              {messages.length - 1} messages
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
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
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                  {message.content}
                </div>
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
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-6">
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
          <p className="text-xs text-gray-500 mt-2 text-center">
            {useRealAPI ? 
              'AI Mode: Connected to Azure OpenAI for advanced responses. Ensure backend API is running.' :
              'Demo Mode: Using pre-configured Belgian ESG scenarios and regulations.'
            }
          </p>
        </div>
      </div>
    </div>
  );
};

export default ESGAgentPageNew;
