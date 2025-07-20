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
      return `### Class F Property Analysis - Brussels (1960 House)

#### Current Market Impact

| Metric | Impact |
|--------|---------|
| Price depreciation vs Class C | -15% to -20% |
| Estimated market reduction | -30 000 – 40 000 € |
| Rental market access | Limited and declining |

#### Regulatory Timeline & Risks

| Year | Regulation | Impact |
|------|-----------|---------|
| **2026** | Mandatory energy audit for rentals | Compliance cost |
| **2028** | Ban on new Class F lease agreements | Rental income loss |
| **2030** | Complete ban on F & G rentals | Total rental prohibition |
| **2035** | Probable extension to Class E | Further restrictions |

#### Renovation Costs & Grants (Brussels)

| Intervention Type | Cost Range (€) | Available Grant |
|-------------------|----------------|-----------------|
| Roof insulation | 8 000 – 12 000 € | Up to 15 €/m² |
| Wall insulation | 10 000 – 15 000 € | Up to 40 €/m² |
| High-efficiency windows | 8 000 – 12 000 € | Up to 25 €/m² |
| Heat pump installation | 12 000 – 18 000 € | 2 000 – 4 000 € |
| **Total renovation cost** | **45 000 – 60 000 €** | **Total grants: 15 000 – 25 000 €** |

#### Financial Return Analysis

| Component | Amount (€) |
|-----------|------------|
| **Total renovation cost** | 45 000 – 60 000 € |
| **Available grants** | -15 000 – 25 000 € |
| **Net investment** | 25 000 – 40 000 € |
| **Property value increase** | +50 000 – 70 000 € |
| **Net ROI** | **+25 000 – 30 000 €** |

#### Action Plan & Timeline
1. **2025**: Schedule EPC+ energy audit
2. **2026**: Apply for renovation grants
3. **2027**: Execute renovation works
4. **2028**: Achieve Class C/B rating

#### Critical Recommendation
**Plan renovation before 2028** to:
- Maximize available grants
- Avoid rental income loss
- Optimize property value recovery
- Ensure regulatory compliance`;
    }
    
    if (lowerQuery.includes('grant') || lowerQuery.includes('subsidy') || lowerQuery.includes('wallonia')) {
      return `### Wallonia Insulation Grants 2025

#### Housing Renovation Grants

| Type d'isolation | Montant (€/m²) | Maximum (€) |
|------------------|----------------|-------------|
| Roof insulation | 15 – 30 €/m² | 3 000 € |
| Wall insulation | 25 – 50 €/m² | 5 000 € |
| Floor insulation | 10 – 20 €/m² | 2 000 € |
| High-performance windows | 15 – 30 €/m² | 2 500 € |

#### Heating System Grants

| System Type | Grant Amount |
|-------------|--------------|
| Air/water heat pump | 2 000 – 4 000 € |
| Geothermal heat pump | 4 000 – 6 000 € |
| Biomass boiler | 1 500 – 3 000 € |
| Solar water heater | 1 000 – 2 000 € |

#### Eligibility Conditions
- **Income limits**: Varies by municipality
- **Mandatory energy audit** before application
- **Certified contractors** only
- **Performance gain**: Minimum 1 EPC class improvement

#### Financial Advantages Combination
- Regional + municipal grants
- **6% VAT reduction** (renovation works)
- **30% tax deduction** (max 3 830 €/year)

**Total cumulative grants possible: 15 000 – 30 000 €** depending on project scope!

#### Next Steps
1. Contact your municipality for specific income thresholds
2. Schedule certified energy audit
3. Get quotes from certified contractors
4. Submit grant application before starting works`;
    }
    
    if (lowerQuery.includes('pump') || lowerQuery.includes('roi') || lowerQuery.includes('heat')) {
      return `### Heat Pump vs Gas ROI Analysis (100m²)

#### Initial Investment Breakdown

| Component | Cost Range (€) |
|-----------|----------------|
| Air/water heat pump | 12 000 – 18 000 € |
| Installation + adaptation | 3 000 – 5 000 € |
| **Total investment** | **15 000 – 23 000 €** |
| Available grants | -3 000 € |
| **Net cost** | **12 000 – 20 000 €** |

#### Annual Operating Costs (100m²)

| Energy Source | Annual Cost | CO₂ Impact |
|---------------|-------------|------------|
| Current gas heating | 1 200 – 1 500 € | High |
| Heat pump system | 800 – 1 000 € | Low |
| **Annual savings** | **400 – 500 €** | **-2.5 tons CO₂/year** |

#### Financial Return Analysis

| Metric | Value |
|--------|--------|
| **20-year energy savings** | 8 000 – 10 000 € |
| **Property value increase** | +10 000 – 15 000 € |
| **Total ROI** | **18 000 – 25 000 €** |
| **Payback period** | **8 – 12 years** |

#### Energy Price Evolution Impact
- **Gas prices**: +3-5% annual increase projected
- **Renewable electricity**: Price stability expected
- **Improved ROI over time** due to price differential

#### EPC Class Improvement
- **Before**: Class E typical
- **After**: Class B/A achievable
- **2030+ compliance**: Guaranteed

#### Verdict
**Profitable long-term investment**, especially considering:
- Available grants reducing initial cost
- 2030 energy performance requirements
- Rising fossil fuel costs
- Property value enhancement`;
    }
    
    if (lowerQuery.includes('2030') || lowerQuery.includes('deadline') || lowerQuery.includes('rental')) {
      return `### 2030 Rental Regulations: Class G Properties

#### Regulatory Timeline

| Year | Regulation | Status |
|------|-----------|---------|
| **2026** | Mandatory energy audit for all rentals | **Upcoming** |
| **2028** | Ban on new Class G lease agreements | **Critical** |
| **2030** | Total ban on F & G rental properties | **Final deadline** |
| **2035** | Probable extension to Class E | **Under consideration** |

#### Immediate Legal & Financial Consequences

| Consequence | Impact |
|-------------|---------|
| **Legal rental prohibition** | Cannot rent legally after 2030 |
| **Financial penalties** | 500 – 2 000 € fines |
| **Complete income loss** | 100% rental revenue elimination |
| **Property depreciation** | -30% to -40% market value |

#### Financial Impact Example (200k€ Property)

| Scenario | Value (€) |
|----------|-----------|
| **Current property value** | 200 000 € |
| **Post-2030 residual value** | 120 000 – 140 000 € |
| **Net financial loss** | **60 000 – 80 000 €** |
| **Avoidable renovation cost** | 40 000 – 50 000 € |

#### Urgent Action Timeline

| Period | Critical Actions |
|---------|------------------|
| **End 2025** | Complete EPC+ energy audit |
| **2026-2027** | Renovation planning & grant applications |
| **End 2027** | Secure contractor & finalize grants |
| **2028** | Complete renovation before rental ban |

#### Time Sensitivity Factors
- **Remaining time**: Only 3-4 years
- **Grant availability**: Reduced after 2027
- **Contractor availability**: Saturated market near 2030
- **Cost inflation**: Higher prices due to demand surge

#### Critical Decision Point
**IMMEDIATE ACTION REQUIRED:**
- **Option A**: Plan & execute renovation now
- **Option B**: Sell before significant depreciation

**Financial Reality**: The longer you wait, the more expensive it becomes and the fewer options remain available.`;
    }

    // Default response
    if (lowerQuery.includes('cost') || lowerQuery.includes('renovation') || lowerQuery.includes('upgrade')) {
      return `### Typical Renovation Costs Breakdown

| Upgrade Type | Estimated Cost Range (€) |
|--------------|---------------------------|
| Roof insulation | 8 000 – 15 000 € |
| Wall insulation | 7 000 – 18 000 € |
| Floor insulation | 4 000 – 8 000 € |
| High-efficiency glazing/windows | 8 000 – 16 000 € |
| Efficient heating system (heat pump/condensing boiler) | 8 000 – 22 000 € |
| Ventilation system | 4 000 – 8 000 € |
| Solar panels (optional/extra) | 5 000 – 8 000 € |

**Total estimate:**  
**35 000 – 80 000 €**  
*(For apartments: 15 000 – 40 000 €)*

#### Factors Influencing Cost
- **Current state**: Poorer condition = higher costs
- **Region**: Brussels, Flanders, and Wallonia have different labor/material costs and grant schemes
- **Depth of renovation**: Partial vs. full envelope & system upgrade

#### Financial Incentives by Region

| Region | Grant Program | Subsidy Rate |
|--------|---------------|--------------|
| **Flanders** | Mijn VerbouwPremie | Up to 50% subsidy, capped per intervention |
| **Wallonia** | Primes Habitation | Sliding scale based on income, up to 70% for low-income |
| **Brussels** | Renolution grants | Various categories, up to 50% |

#### Practical Advice
- **Start with an EPC+ renovation plan**: A certified energy advisor can prioritize cost-effective actions and help you sequence works for maximum subsidy
- **Combine grants**: You can stack different regional and federal incentives
- **Check local requirements**: Some municipalities provide additional support or tax reductions

**Tip**: Always request multiple quotes and plan works to maximize grant eligibility (some require pre-approval).

Let me know your region and property type for a more tailored estimate or a grant simulation.`;
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

  // Fonction pour convertir le markdown en HTML (version exacte du SidePanel qui fonctionne)
  const formatMessageText = (text) => {
    if (!text) return { __html: '' };
    
    let formattedText = text.toString().trim();
    
    // Vérifier si le message est vide ou ne contient que des espaces/caractères spéciaux
    if (!formattedText || formattedText.match(/^\s*[#\s]*$/)) {
      return { __html: '' };
    }
    
    console.log('📝 ESG Agent formatting text:', formattedText.substring(0, 100) + '...');
    
    // 0. SUPPRIMER TOUS LES EMOJIS ET ICÔNES (en premier)
    formattedText = formattedText.replace(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]|[\u{1F900}-\u{1F9FF}]|[\u{1F018}-\u{1F0FF}]/gu, '');
    // Supprimer les caractères spéciaux couramment utilisés comme icônes sauf ceux utiles
    formattedText = formattedText.replace(/[🔄📊🏠💰⚡🌱📈🧠⚠️✅🤖💡👋🏷️📅🔥🏗️ℹ️📋❌🎯]/g, '');
    // Nettoyer les espaces multiples résultant de la suppression d'emojis
    formattedText = formattedText.replace(/\s{2,}/g, ' ');
    
    // 1. SUPPRIMER LES # QUI TRAÎNENT EN FIN DE LIGNE ET LIGNES
    formattedText = formattedText.replace(/^\s*#\s*$/gm, ''); // Supprimer les lignes avec juste #
    formattedText = formattedText.replace(/\s+#\s*$/gm, ''); // Supprimer les # en fin de ligne
    
    // 2. Supprimer les séparateurs markdown (barres horizontales) EN PREMIER
    formattedText = formattedText.replace(/---+/g, '');
    formattedText = formattedText.replace(/^\s*-{3,}\s*$/gm, '');
    formattedText = formattedText.replace(/^\s*\*{3,}\s*$/gm, '');
    formattedText = formattedText.replace(/^\s*_{3,}\s*$/gm, '');
    
    // 3. Gérer les titres avec numéros qui peuvent être collés au texte précédent
    formattedText = formattedText.replace(/(\w+)(\s*#{3}\s*\d+\.\s+[^\n]+)/g, '$1<br/>$2');
    formattedText = formattedText.replace(/(\w+)(\s*#{2}\s*\d+\.\s+[^\n]+)/g, '$1<br/>$2');
    formattedText = formattedText.replace(/(\w+)(\s*#{1}\s*\d+\.\s+[^\n]+)/g, '$1<br/>$2');
    
    // 4. Convertir les titres markdown # ## ### #### en titres HTML propres
    // Ordre important : du plus spécifique au moins spécifique
    // Gérer les cas où les titres peuvent être au milieu du texte ou après des espaces
    formattedText = formattedText.replace(/(^|[\s\n])#{4}\s+([^\n]+)/gm, '$1<h5 style="margin: 10px 0 5px 0; font-weight: 600; color: #1565c0; font-size: 13px;">$2</h5>');
    formattedText = formattedText.replace(/(^|[\s\n])#{3}\s+([^\n]+)/gm, '$1<h4 style="margin: 12px 0 6px 0; font-weight: bold; color: #2563eb; font-size: 14px;">$2</h4>');
    formattedText = formattedText.replace(/(^|[\s\n])#{2}\s+([^\n]+)/gm, '$1<h3 style="margin: 14px 0 8px 0; font-weight: bold; color: #8b5a2b; font-size: 15px;">$2</h3>');
    formattedText = formattedText.replace(/(^|[\s\n])#{1}\s+([^\n]+)/gm, '$1<h2 style="margin: 16px 0 10px 0; font-weight: bold; color: #6a1b9a; font-size: 17px; border-bottom: 1px solid #e0e0e0; padding-bottom: 3px;">$2</h2>');
    
    // 4.1 Gérer les titres avec numéros qui peuvent être au milieu du texte
    formattedText = formattedText.replace(/(?:^|\n)#{3}\s*(\d+)\.\s+([^\n]+)/gm, '<h4 style="margin: 12px 0 6px 0; font-weight: bold; color: #2563eb; font-size: 14px;">$1. $2</h4>');
    formattedText = formattedText.replace(/(?:^|\n)#{2}\s*(\d+)\.\s+([^\n]+)/gm, '<h3 style="margin: 14px 0 8px 0; font-weight: bold; color: #8b5a2b; font-size: 15px;">$1. $2</h3>');
    formattedText = formattedText.replace(/(?:^|\n)#{1}\s*(\d+)\.\s+([^\n]+)/gm, '<h2 style="margin: 16px 0 10px 0; font-weight: bold; color: #6a1b9a; font-size: 17px; border-bottom: 1px solid #e0e0e0; padding-bottom: 3px;">$1. $2</h2>');
    
    // 5. FORMATAGE SPÉCIALISÉ POUR LES TAGS *Risk:* et *Mitigation:* (AVANT les autres formattages)
    // Convertir *Risk:* en span rouge sans icône
    formattedText = formattedText.replace(/\*Risk:\*/g, '<span style="color: #d32f2f; font-weight: bold; background-color: #ffebee; padding: 2px 6px; border-radius: 3px; font-size: 12px;">RISK</span>');
    // Convertir *Mitigation:* en span vert sans icône
    formattedText = formattedText.replace(/\*Mitigation:\*/g, '<span style="color: #2e7d32; font-weight: bold; background-color: #e8f5e8; padding: 2px 6px; border-radius: 3px; font-size: 12px;">MITIGATION</span>');
    
    // 6. CONVERTIR LES TABLES MARKDOWN EN HTML (avant les autres conversions)
    // Pattern simple pour détecter les tables markdown
    formattedText = formattedText.replace(/(\|[^\n]+\|\n\|[-\s|:]*\|\n(?:\|[^\n]*\|\n?)*)/g, (match) => {
      console.log('📊 Found table, converting...', match.substring(0, 100));
      
      const lines = match.trim().split('\n').filter(line => line.trim());
      if (lines.length < 2) return match;
      
      // Première ligne = headers
      const headerLine = lines[0];
      const headers = headerLine.split('|').map(h => h.trim()).filter(h => h);
      
      // Deuxième ligne = séparateur (on l'ignore)
      
      // Lignes suivantes = données
      const dataLines = lines.slice(2);
      const rows = dataLines.map(line => 
        line.split('|').map(cell => cell.trim()).filter(cell => cell !== '')
      ).filter(row => row.length > 0);
      
      // Générer la table HTML
      let tableHTML = '<table style="width: 100%; border-collapse: collapse; border: 1px solid #ddd; margin: 15px 0; font-size: 14px;">';
      
      // Headers
      tableHTML += '<thead style="background-color: #f8f9fa;"><tr>';
      headers.forEach(header => {
        const formattedHeader = header.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        tableHTML += `<th style="border: 1px solid #ddd; padding: 10px; text-align: left; font-weight: bold; color: #333;">${formattedHeader}</th>`;
      });
      tableHTML += '</tr></thead>';
      
      // Body
      tableHTML += '<tbody>';
      rows.forEach(row => {
        tableHTML += '<tr>';
        row.forEach(cell => {
          // Convertir **texte** en gras dans les cellules
          const formattedCell = cell.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
          tableHTML += `<td style="border: 1px solid #ddd; padding: 8px; vertical-align: top;">${formattedCell}</td>`;
        });
        tableHTML += '</tr>';
      });
      tableHTML += '</tbody></table>';
      
      return tableHTML;
    });
    
    // 7. Convertir **texte** en <strong>texte</strong> APRÈS les titres et tags spécialisés pour éviter les conflits
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 8. Traiter les puces différemment selon leur contexte
    // D'abord traiter les puces qui suivent "How:" ou "Why:"
    formattedText = formattedText.replace(/(Why:|How:)\s*\n?\s*-\s*(.+)/gi, '$1<br/><div style="margin: 4px 0 4px 16px; padding: 0; line-height: 1.4;"><span style="color: #6a1b9a; font-weight: bold;">•</span> $2</div>');
    
    // Ensuite traiter les puces normales en début de ligne
    formattedText = formattedText.replace(/^[•\+\-]\s*(.*?)$/gm, '<div style="margin: 2px 0 2px 16px; padding: 0; line-height: 1.4; position: relative;"><span style="position: absolute; left: -12px; color: #6a1b9a; font-weight: bold;">•</span>$1</div>');
    
    // 9. Nettoyer les sauts de ligne excessifs
    formattedText = formattedText.replace(/\n\s*\n\s*\n/g, '\n\n');
    
    // 10. Convertir les sauts de ligne simples en <br/> mais éviter autour des balises HTML
    formattedText = formattedText.replace(/\n(?!\s*<)/g, '<br/>');
    
    // 11. Nettoyer les <br/> en trop et les lignes vides
    formattedText = formattedText.replace(/(<br\/>){3,}/g, '<br/><br/>');
    formattedText = formattedText.replace(/<br\/>\s*(<h[1-6])/g, '$1');
    formattedText = formattedText.replace(/(<\/h[1-6]>)\s*<br\/>/g, '$1');
    formattedText = formattedText.replace(/<br\/>\s*(<div)/g, '$1');
    formattedText = formattedText.replace(/(<\/div>)\s*<br\/>/g, '$1');
    formattedText = formattedText.replace(/<br\/>\s*(<table)/g, '$1');
    formattedText = formattedText.replace(/(<\/table>)\s*<br\/>/g, '$1');
    
    // 12. Supprimer les lignes vides restantes
    formattedText = formattedText.replace(/<br\/>\s*<br\/>\s*<br\/>/g, '<br/><br/>');
    
    // 13. Nettoyer les espaces en début et fin
    formattedText = formattedText.trim();
    
    console.log('✅ ESG Agent formatted result:', formattedText.substring(0, 200) + '...');
    
    // Dernière vérification : si le contenu final est vide, retourner vide
    const finalContent = formattedText.replace(/<[^>]*>/g, '').trim();
    if (!finalContent) {
      return { __html: '' };
    }
    
    return { __html: formattedText };
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
          <button
            onClick={() => {
              const testMarkdown = `### Test Table
              
| Metric | Value |
|--------|-------|
| **Test** | Success |
| Price | €100,000 |

**This is bold text**
- Bullet point 1
- Bullet point 2`;
              setMessages(prev => [...prev, { type: 'agent', content: testMarkdown }]);
            }}
            className="bg-red-100 border border-red-300 text-red-700 px-4 py-2 rounded-lg text-sm hover:bg-red-200 transition-all duration-200"
            title="Test markdown formatting"
          >
            🧪 Test Markdown
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="h-[600px] overflow-y-auto p-6 space-y-6 bg-gray-50">
        {messages.map((message, index) => {
          console.log('🔍 BelgianESGAgent rendering message', index, ':', {
            type: message.type,
            contentPreview: message.content?.substring(0, 100) + '...',
            messageStructure: Object.keys(message)
          });
          
          return (
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
                <div 
                  className="text-sm leading-relaxed"
                  dangerouslySetInnerHTML={formatMessageText(message.content)}
                />
              </div>
            </div>
          );
        })}
        
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
