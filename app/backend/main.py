"""
Main FastAPI application for Real Estate Price Predictor
Includes ESG chat endpoint and training jobs management
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import os

# Import training jobs router
from api.training_jobs import router as training_jobs_router

# Create FastAPI app
app = FastAPI(
    title="Real Estate AI Platform API",
    description="API for Real Estate Price Prediction and ESG Advisory",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training_jobs_router)

# Chat request model
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

# ESG Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    ESG Real Estate Chat Endpoint
    Provides AI-powered advice for Belgian real estate ESG compliance
    """
    try:
        # Extract user message
        user_message = request.messages[-1].content if request.messages else ""
        
        # Simple ESG response generator (replace with Azure OpenAI in production)
        response = generate_esg_response(user_message)
        
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

def generate_esg_response(query: str) -> str:
    """Generate contextual ESG response based on user query"""
    
    query_lower = query.lower()
    
    # EPC Class queries
    if 'class f' in query_lower or ('epc' in query_lower and any(word in query_lower for word in ['class', 'f', 'g'])):
        return """EPC Class F Properties - Investment Risk & Strategy

**Immediate Risks (2024-2026):**
- Limited rental market access
- Difficulty obtaining mortgages
- Potential value depreciation

**Regulatory Timeline:**
- 2026: Energy audit mandatory for all rentals
- 2030: Class F properties cannot be rented
- 2035: Class G properties banned from rental market

**Investment Strategy:**
1. Purchase at discount (20-30% below market)
2. Budget €15,000-35,000 for EPC upgrades
3. Target Class D minimum (rental viable until 2030+)

**Priority Renovations:**
- Roof insulation: €3,000-8,000 (highest ROI)
- Wall insulation: €8,000-15,000
- High-efficiency heating: €8,000-12,000
- Windows upgrade: €400-800 per m²

**Available Grants:**
- Flanders: MyEnergyPremium up to €2,500
- Wallonia: Prime Habitation up to €5,000
- Brussels: Renolution premium up to €35,000"""

    # Grant queries
    elif any(word in query_lower for word in ['grant', 'subsidy', 'premium', 'aide']):
        return """Belgian Renovation Grants & Subsidies 2024

**Flanders - MyEnergyPremium:**
- Roof insulation: €6/m² (max €1,500)
- Wall insulation: €8/m² (max €2,500)
- Heat pump: €800-2,500 depending on type
- Solar panels: €300 per kWp (max €2,400)

**Wallonia - Prime Habitation:**
- Income-based rates (€1,000-5,000)
- Insulation: 30-50% of costs covered
- Heat pump installation: €1,000-3,000
- Energy audit: €500 reimbursement

**Brussels - Renolution:**
- Comprehensive renovations: up to €35,000
- Single measures: €500-8,000
- Low-income households: up to 80% coverage
- Energy audit: 100% covered

**Application Process:**
1. Energy audit (mandatory for most grants)
2. Submit application before starting work
3. Use certified contractors
4. Keep all receipts and documentation
5. Final inspection required

**Tax Benefits:**
- 45% tax reduction on energy improvements
- Maximum €3,200 per year
- Valid through 2024"""

    # 2030 compliance queries
    elif '2030' in query_lower or 'deadline' in query_lower or 'compliance' in query_lower:
        return """2030-2035 ESG Compliance Deadlines

**2030 Rental Restrictions:**
- EPC Class F properties: rental ban begins
- Energy audit mandatory for all rentals
- Minimum energy requirements for new leases

**2035 Complete Ban:**
- EPC Class G properties: cannot be rented
- Stricter energy performance standards
- Enhanced penalties for non-compliance

**Compliance Strategy Timeline:**
**2024-2025: Assessment Phase**
- Conduct energy audits for all properties
- Prioritize worst-performing assets
- Secure financing and grants

**2026-2028: Implementation Phase**
- Execute major renovations
- Target Class D minimum performance
- Monitor regulatory updates

**2029-2030: Final Compliance**
- Complete remaining improvements
- Verify EPC certifications
- Prepare for stricter 2035 standards

**Investment Priorities:**
1. **Immediate ROI**: Insulation, heating efficiency
2. **Future-Proofing**: Heat pumps, smart systems
3. **Market Positioning**: Class C or better

**Financial Planning:**
- Budget €20,000-40,000 per property
- Use available grants (can cover 30-50%)
- Consider green financing options
- Factor in rental premiums for efficient properties"""

    # General ESG advice
    else:
        return f"""Belgian Real Estate ESG Advisory

Thank you for your question: "{query}"

As your ESG advisor, I recommend focusing on these key areas:

**Energy Performance Priorities:**
- Upgrade heating systems to high-efficiency options
- Improve insulation (roof, walls, windows)
- Consider renewable energy installation
- Target EPC Class D minimum for future rental viability

**Regulatory Compliance:**
- 2030: EPC Class F rental restrictions begin
- Energy audits becoming mandatory
- Document all improvements for compliance

**Financial Strategy:**
- Leverage available grants (Flanders, Wallonia, Brussels)
- Factor in 45% tax reduction for energy improvements
- Budget for comprehensive renovations: €20,000-40,000 per property

**Market Positioning:**
- Energy-efficient properties command rental premiums
- Better financing terms for sustainable investments
- Enhanced property values in ESG-conscious market

Would you like specific advice on any of these areas? I can provide detailed guidance on grants, renovation priorities, or compliance strategies for your particular situation."""

    return response

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Real Estate AI Platform API",
        "status": "healthy",
        "endpoints": {
            "chat": "/api/chat",
            "training_jobs": "/api/training-jobs",
            "health": "/api/training-jobs/health"
        }
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
