# AdminPanel Integration Guide

## 🎯 Overview
The AdminPanel is a comprehensive debugging and monitoring interface for LLM prompt visualization and system management. It provides real-time capture and analysis of all prompts sent to the LLM system.

## 🚀 Features

### 1. Advanced UI Capabilities
- **Drag & Drop**: Detach and reposition the panel anywhere on screen
- **Multi-directional Resize**: Resize from any edge or corner
- **Tab-based Interface**: 5 organized tabs for different functions
- **Responsive Design**: Adapts to different screen sizes

### 2. Prompt Visualization
- **Real-time Capture**: Automatically captures all LLM prompts
- **Categorized Display**: Different prompt types (ESG Analysis, Strategic Analysis, Fallback)
- **Metadata Information**: Timestamps, locations, scores, and context
- **Management Tools**: Copy, export, and clear prompt history

### 3. System Monitoring
- **Live Metrics**: Real-time system performance data
- **Error Tracking**: Comprehensive error logging and analysis
- **Configuration Management**: System settings and parameters
- **Data Analytics**: Usage patterns and performance insights

## 📋 How to Use

### Opening the AdminPanel
1. Click the **Admin** button in the global navigation menu
2. The panel will slide in from the right side of the screen
3. Use the **Detach** button to create a floating window

### Prompt Visualization Tab
1. Navigate to the **"Prompt Visualization"** tab
2. The panel will automatically capture prompts as they are sent to the LLM
3. Each prompt shows:
   - **Type**: ESG_ANALYSIS, STRATEGIC_ANALYSIS, or ESG_ANALYSIS_FALLBACK
   - **Timestamp**: When the prompt was sent
   - **Content**: Full prompt text with syntax highlighting
   - **Metadata**: Additional context and parameters

### Management Features
- **Copy**: Click the copy icon to copy a prompt to clipboard
- **Export**: Export all prompts as JSON for analysis
- **Clear**: Clear all captured prompts
- **Filter**: Search and filter prompts by type or content

## 🧪 Testing

### Manual Testing
1. Open the AdminPanel and navigate to "Prompt Visualization"
2. Fill out a property form and submit for ESG analysis
3. Check that prompts appear in real-time in the AdminPanel
4. Test the drag/drop and resize functionality

### Automated Testing
Run the test script in the browser console:
```javascript
testAdminPanelPromptCapture()
```

This will simulate prompt dispatching and populate the AdminPanel with test data.

## 🔧 Technical Implementation

### Event-Driven Architecture
The AdminPanel uses custom events to capture prompts:
```javascript
window.dispatchEvent(new CustomEvent('llmPromptSent', {
  detail: {
    type: 'ESG_ANALYSIS',
    prompt: promptText,
    timestamp: timestamp,
    metadata: { /* additional context */ }
  }
}));
```

### Prompt Capture Integration
Prompts are automatically captured in `PropertyForm.js` at three key points:
1. **ESG Analysis**: When ESG evaluation is performed
2. **Strategic Analysis**: When strategic analysis is generated
3. **Fallback Analysis**: When API fails and fallback is used

### Data Structure
```javascript
{
  type: 'ESG_ANALYSIS' | 'STRATEGIC_ANALYSIS' | 'ESG_ANALYSIS_FALLBACK',
  prompt: string,
  timestamp: string,
  metadata: {
    esgScores?: object,
    location?: string,
    postalCode?: string,
    propertyType?: string,
    fallbackReason?: string
  }
}
```

## 🎨 UI Features

### Drag & Drop
- Click and drag the header to move the detached panel
- Smooth animations and positioning
- Automatic boundary detection

### Resize Functionality
- **Edges**: Left, right, top, bottom
- **Corners**: All four corners for diagonal resizing
- **Visual Feedback**: Hover effects and cursor changes
- **Constraints**: Minimum and maximum size limits

### Styling
- **Gradient Background**: Purple gradient with transparency
- **Modern Design**: Clean, professional interface
- **Responsive Layout**: Adapts to different screen sizes
- **Accessibility**: ARIA labels and keyboard navigation

## 🔍 Debugging

### Common Issues
1. **Prompts not appearing**: Check that events are being dispatched
2. **Styling issues**: Verify CSS file is imported correctly
3. **Resize not working**: Check mouse event handlers
4. **Performance**: Monitor for memory leaks with event listeners

### Debug Console
Check the browser console for:
- Prompt dispatch events
- AdminPanel state changes
- Error messages and warnings
- Performance metrics

## 📊 Data Flow

```
PropertyForm.js → Event Dispatch → AdminPanel.jsx → State Update → UI Render
     ↓                    ↓              ↓              ↓           ↓
  LLM Prompt    CustomEvent('llmPromptSent')    Capture    Display   User Interface
```

## 🛠️ Future Enhancements

1. **Export Formats**: PDF, CSV, XML export options
2. **Advanced Filtering**: Date ranges, prompt types, content search
3. **Analytics Dashboard**: Usage statistics and performance metrics
4. **Integration**: Connect with external monitoring tools
5. **Collaboration**: Share prompts and analysis with team members

## 📞 Support

For issues or questions:
1. Check the browser console for error messages
2. Verify all files are correctly imported
3. Test with the provided test script
4. Review the implementation in `PropertyForm.js` and `AdminPanel.jsx`
