# ESG Data Transmission Fix

## Problem Identified
The ESG scores displayed in the central panel (Environmental: 10/10, Social: 8.8/10, Governance: 8/10, Overall ESG: 8.9/10) were not being correctly transmitted to the left SidePanel for strategic analysis. The SidePanel was showing default values (Environmental: 7/10, Social: 7/10, Governance: 7/10) instead of the actual calculated scores.

## Root Cause
The issue was in the field mapping between the `EsgSummary` component and the `SidePanel` component:

- **EsgSummary** stores scores with keys: `environment`, `social`, `governance`, `overall`
- **SidePanel** was trying to access scores with keys: `environmental`, `social`, `governance`, `overall`

## Solution Applied

### 1. Added Debug Logging
```jsx
console.log("ESG Data received:", esgData);
console.log("ESG Scores extracted:", esgData?.esg_scores);
console.log("Final ESG Scores in analysisData:", analysisData.esgScores);
```

### 2. Fixed Field Mapping
Updated the strategic analysis prompt in `SidePanel.jsx` to use the correct field names:

**Before:**
```jsx
Based on ESG scores (Environmental: ${analysisData.esgScores.environmental || 'N/A'}/10, Social: ${analysisData.esgScores.social || 'N/A'}/10, Governance: ${analysisData.esgScores.governance || 'N/A'}/10), analyze the investment potential.
```

**After:**
```jsx
Based on ESG scores (Environmental: ${analysisData.esgScores.environment || analysisData.esgScores.environmental || 'N/A'}/10, Social: ${analysisData.esgScores.social || 'N/A'}/10, Governance: ${analysisData.esgScores.governance || 'N/A'}/10, Overall: ${analysisData.esgScores.overall || 'N/A'}/10), analyze the investment potential.
```

### 3. Added Fallback Support
The fix includes fallback support for both field naming conventions to ensure compatibility.

## Expected Result
The strategic analysis in the SidePanel should now display the correct ESG scores:
- Environmental: 10/10 (instead of 7/10)
- Social: 8.8/10 (instead of 7/10) 
- Governance: 8/10 (instead of 7/10)
- Overall: 8.9/10 (new addition)

## Files Modified
- `app/frontend-react/src/components/SidePanel/SidePanel.jsx`

## Testing
1. Fill out the property form
2. Generate ESG analysis
3. Click "ESG Strategic Analysis" in the SidePanel
4. Verify that the "Investment Positioning" section shows the correct ESG scores matching the central panel

## Status
✅ Fix applied and ready for testing
