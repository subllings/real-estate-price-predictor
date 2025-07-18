// Test script to verify AdminPanel integration
// This can be run in the browser console to test the LLM prompt capture

// Test function to simulate LLM prompt dispatch
function testAdminPanelPromptCapture() {
  console.log('🧪 Testing AdminPanel prompt capture...');
  
  // Simulate ESG Analysis prompt
  window.dispatchEvent(new CustomEvent('llmPromptSent', {
    detail: {
      type: 'ESG_ANALYSIS',
      prompt: 'ESG Scores - Environmental: 8/10, Social: 7/10, Governance: 9/10, Overall: 8/10 | Apartment in 1050 Ixelles, Brussels-Capital Region (3:45:21 PM)',
      timestamp: '3:45:21 PM',
      metadata: {
        esgScoresIncluded: 'ESG Scores - Environmental: 8/10, Social: 7/10, Governance: 9/10, Overall: 8/10',
        calculatedScores: {
          environmental: 8,
          social: 7,
          governance: 9,
          overall: 8
        },
        location: '1050 Ixelles',
        postalCode: '1050'
      }
    }
  }));
  
  // Simulate Strategic Analysis prompt
  setTimeout(() => {
    window.dispatchEvent(new CustomEvent('llmPromptSent', {
      detail: {
        type: 'STRATEGIC_ANALYSIS',
        prompt: `Generate a comprehensive strategic analysis for this Belgian real estate investment...
        
# Strategic Analysis – 1050 Ixelles Property Investment

## ESG Analysis Summary
**Environmental Score:** 8/10 **Social Score:** 7/10 **Governance Score:** 9/10 **Overall ESG Score:** 8/10

## Investment Positioning
Based on the ESG analysis results above, analyze the investment potential for this apartment property in 1050 Ixelles...`,
        timestamp: '3:45:25 PM',
        metadata: {
          esgScores: {
            environmental: 8,
            social: 7,
            governance: 9,
            overall: 8
          },
          location: '1050 Ixelles',
          postalCode: '1050',
          propertyType: 'Apartment',
          surface: 120,
          bedrooms: 3
        }
      }
    }));
  }, 2000);
  
  // Simulate fallback prompt
  setTimeout(() => {
    window.dispatchEvent(new CustomEvent('llmPromptSent', {
      detail: {
        type: 'ESG_ANALYSIS_FALLBACK',
        prompt: 'ESG Scores - Environmental: 6/10, Social: 5/10, Governance: 7/10, Overall: 6/10 | House in 1000 Brussels, Brussels-Capital Region (3:45:30 PM) [FALLBACK]',
        timestamp: '3:45:30 PM',
        metadata: {
          esgScoresIncluded: 'ESG Scores - Environmental: 6/10, Social: 5/10, Governance: 7/10, Overall: 6/10',
          calculatedScores: {
            environmental: 6,
            social: 5,
            governance: 7,
            overall: 6
          },
          location: '1000 Brussels',
          postalCode: '1000',
          fallbackReason: 'API unavailable'
        }
      }
    }));
  }, 4000);
  
  console.log('✅ Test prompts dispatched! Check the AdminPanel for captured prompts.');
}

// Export function for browser console usage
window.testAdminPanelPromptCapture = testAdminPanelPromptCapture;

console.log('🚀 AdminPanel test script loaded!');
console.log('📋 To test: Open AdminPanel and run testAdminPanelPromptCapture() in console');
