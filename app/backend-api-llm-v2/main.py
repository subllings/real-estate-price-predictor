import os
import re
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import json
import logging

logger = logging.getLogger("uvicorn.error")

# Helper function to correctly format EPC scores
def format_epc_score(epc_score):
    """Convert EPC score format correctly, handling A_plus -> A+ specifically"""
    if epc_score == 'A_plus':
        return 'A+'
    return epc_score.replace('_', '+') if epc_score else 'N/A'

# Load environment variables
load_dotenv()

# FastAPI instance
app = FastAPI(
    title="Azure OpenAI API",
    description="Unified API to interact with Azure OpenAI using FastAPI",
    version="1.0.0"
)

# CORS config
origins = [
    "https://realestate-react-ui-agent.azurewebsites.net",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# === Environment Config ===
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# === Schemas ===

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
     messages: List[Message]

class ChatResponse(BaseModel):
    response: str

class CommentResponse(BaseModel):
    comments: List[str]

class ScoreMeta(BaseModel):
    mae: float
    rmse: float
    r2: float

class FormData(BaseModel):
    region: str
    province: str
    locality: str
    scoreMeta: ScoreMeta

class UserProfile(BaseModel):
    type: str
    objectives: List[str]
    language: str

class CommentRequest(BaseModel):
    formData: FormData
    predictionAll: float
    predictionTop: float
    userProfile: UserProfile

class LLMParamRequest(BaseModel):
    model_name: str

class SuggestionRequest(BaseModel):
    model_name: str
    previous_trials: list

class PropertyFeatures(BaseModel):
    propertyType: str
    subtype: str
    province: str
    locality: str
    postCode: str
    bedroomCount: int
    bathroomCount: int
    toiletCount: int
    roomCount: int
    habitableSurface: int
    facedeCount: int
    buildingConstructionYear: int
    buildingCondition: str
    kitchenType: str
    heatingType: str
    floodZoneType: str
    epcScore: str
    hasLivingRoom: bool
    hasTerrace: bool

class ESGAnalysisRequest(BaseModel):
    propertyFeatures: PropertyFeatures
    estimatedPrice: float
    analysis_depth: str = "detailed"  # "basic", "detailed", "comprehensive"

class ESGAnalysisResponse(BaseModel):
    analysis_points: List[str]
    esg_scores: dict
    recommendations: List[str]
    compliance_status: dict
    financial_impact: dict
    full_report: str

class AgentInsight(BaseModel):
    agent: str
    summary: str

class ESGSummary(BaseModel):
    environment: float
    social: float
    governance: float
    overall: str

class StrategicSummaryRequest(BaseModel):
    price_prediction: float
    esg_summary: ESGSummary
    property_features: PropertyFeatures
    strategic_goals: str  # "invest", "live", "renovate", etc.
    agent_insights: List[AgentInsight]

class StrategicSummaryResponse(BaseModel):
    strategic_positioning: str
    esg_analysis: str
    recommended_actions: str
    clickable_suggestions: List[dict]
    confidence_score: float

# === Utility Function ===

def call_azure_openai_chat(messages: List[dict], temperature: float = 0.7, max_tokens: int = 1500):
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        print("Payload sent to Azure OpenAI:", payload)  
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Azure OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from Azure OpenAI.")

# === Routes ===


@app.get("/", tags=["Health"])
def root():
    return {"message": "API LLM V2 is running..."}

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Missing messages field")

    response_text = call_azure_openai_chat(
        messages=[msg.dict() for msg in request.messages]
    )
    return {"response": response_text}

@app.post("/comment")
def generate_comments(request: CommentRequest):
    form = request.formData
    profile = request.userProfile

    prompt = f"""
    You are a real estate investment advisor for a user profile of type '{profile.type}', 
    with objectives: {', '.join(profile.objectives)}. 
    The property is located in {form.locality}, {form.province}, {form.region}.
    
    Predicted property value: {request.predictionAll:,.0f} € (use European format: space as thousand separator, € after amount)
    Model performance: MAE = {form.scoreMeta.mae:,.0f} €, RMSE = {form.scoreMeta.rmse:,.0f} €, R² = {form.scoreMeta.r2:.2f}

    Focus on:
    1. Regional market trends and investment potential for {form.region}
    2. Property characteristics analysis and investment return strategy
    3. Market insights specific to {form.locality}

    IMPORTANT: When mentioning prices or amounts, always use European format: "417 675 €" (space as thousand separator, € symbol after the amount), NOT "€417,675".

    Generate 2-3 concise, investment-focused comments in English. Each comment should be practical and actionable for real estate investment decisions.
    """

    response_text = call_azure_openai_chat(
        messages=[
            {"role": "system", "content": "You are a helpful real estate AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    comments = [c.strip() for c in response_text.strip().split('\n') if c.strip()]
    return {"comments": comments}

from fastapi import APIRouter, HTTPException
import json

router = APIRouter()
app.include_router(router)

@router.post("/suggest-space", tags=["LLM Tuner"])
def suggest_param_space(request: SuggestionRequest):
    model_name = request.model_name
    trials = request.previous_trials

    # Build a summary of previous trials for the prompt
    if not trials:
        trial_summary = "(no prior trials found)"
    else:
        trial_summary = "\n".join([
            f"- Params: {t.get('hyperparameters', t.get('params'))}, "
            f"Score: {t.get('r2_test') or t.get('rmse') or '?'}"
            for t in trials
        ])

    # Construct model-specific prompts
    if model_name.lower() == "catboost":
        prompt = f"""
        You are an expert in hyperparameter tuning using Optuna for CatBoost regression models.

        You are optimizing a CatBoostRegressor for real estate price prediction. The model uses CPU processing for stability and we want to reduce overfitting while maintaining good performance.

        Here are previous trials for the model '{model_name}':

        {trial_summary}

        Based on this, suggest a comprehensive Optuna parameter space in JSON format that includes ALL important CatBoost parameters:

        REQUIRED PARAMETERS TO INCLUDE:
        - learning_rate (suggest_loguniform: 0.01-0.3)
        - depth (suggest_int: 4-10) 
        - iterations (suggest_int: 100-2000)
        - l2_leaf_reg (suggest_loguniform: 1.0-10.0)
        - border_count (suggest_int: 32-255)
        - random_strength (suggest_uniform: 0.1-10.0)
        - min_data_in_leaf (suggest_int: 1-20)
        - bootstrap_type (suggest_categorical: ["Bayesian", "Bernoulli", "MVS"])
        - subsample (suggest_uniform: 0.6-1.0) - ONLY if bootstrap_type != "Bayesian"
        - grow_policy (suggest_categorical: ["SymmetricTree", "Depthwise", "Lossguide"])
        - leaf_estimation_method (suggest_categorical: ["Newton", "Gradient"])
        - leaf_estimation_iterations (suggest_int: 1-10)
        - bagging_temperature (suggest_uniform: 0.0-1.0)
        - colsample_bylevel (suggest_uniform: 0.5-1.0)
        - od_type (suggest_categorical: ["IncToDec", "Iter"])
        - od_wait (suggest_int: 10-50)
        - task_type (fixed_value: "CPU")

        IMPORTANT RULES:
        1. Use "method" field to specify suggest_loguniform, suggest_uniform, suggest_int, suggest_categorical, or fixed_value
        2. Include "low" and "high" for numeric parameters
        3. Include "choices" for categorical parameters  
        4. Include "value" for fixed_value parameters
        5. Focus on anti-overfitting: lower learning rates, higher regularization, reasonable depth

        OUTPUT FORMAT:
        {{
            "model": "{model_name}",
            "param_space": {{
                "parameter_name": {{"method": "suggest_type", "low": X, "high": Y}},
                "categorical_param": {{"method": "suggest_categorical", "choices": [...]}}
            }}
        }}

        Only output valid JSON. No markdown, no explanations, no additional text.
        """
    
    elif model_name.lower() == "xgboost":
        prompt = f"""
        You are an expert in hyperparameter tuning using Optuna for XGBoost regression models.

        You are optimizing an XGBRegressor for real estate price prediction. The model uses CPU processing for stability (tree_method='auto') and we want to reduce overfitting while maintaining good performance.

        Here are previous trials for the model '{model_name}':

        {trial_summary}

        Based on this, suggest a comprehensive Optuna parameter space in JSON format that includes ALL important XGBoost parameters:

        REQUIRED PARAMETERS TO INCLUDE:
        - learning_rate (suggest_float: 0.01-0.3)
        - max_depth (suggest_int: 3-10)
        - min_child_weight (suggest_float: 1.0-10.0)
        - subsample (suggest_float: 0.5-1.0)
        - colsample_bytree (suggest_float: 0.5-1.0)
        - colsample_bylevel (suggest_float: 0.5-1.0)
        - colsample_bynode (suggest_float: 0.5-1.0)
        - gamma (suggest_float: 0.0-5.0)
        - reg_alpha (suggest_float: 0.0-1.0)
        - reg_lambda (suggest_float: 0.0-2.0)
        - n_estimators (suggest_int: 100-2000)
        - max_delta_step (suggest_int: 0-10)
        - grow_policy (suggest_categorical: ["depthwise", "lossguide"])
        - max_leaves (suggest_int: 0-256) - ONLY if grow_policy == "lossguide"

        IMPORTANT RULES:
        1. Use "method" field to specify suggest_float, suggest_int, or suggest_categorical
        2. Include "low" and "high" for numeric parameters
        3. Include "choices" for categorical parameters
        4. Focus on anti-overfitting: lower learning rates, higher regularization, reasonable depth
        5. Tree method will be set to 'auto' (CPU) for stability
        6. ALWAYS include n_estimators parameter - it's required!

        OUTPUT FORMAT (MUST include n_estimators):
        {{
            "model": "{model_name}",
            "param_space": {{
                "learning_rate": {{"method": "suggest_float", "low": 0.01, "high": 0.3}},
                "max_depth": {{"method": "suggest_int", "low": 3, "high": 10}},
                "n_estimators": {{"method": "suggest_int", "low": 100, "high": 2000}},
                "min_child_weight": {{"method": "suggest_float", "low": 1.0, "high": 10.0}},
                "subsample": {{"method": "suggest_float", "low": 0.5, "high": 1.0}},
                "colsample_bytree": {{"method": "suggest_float", "low": 0.5, "high": 1.0}},
                "colsample_bylevel": {{"method": "suggest_float", "low": 0.5, "high": 1.0}},
                "colsample_bynode": {{"method": "suggest_float", "low": 0.5, "high": 1.0}},
                "gamma": {{"method": "suggest_float", "low": 0.0, "high": 5.0}},
                "reg_alpha": {{"method": "suggest_float", "low": 0.0, "high": 1.0}},
                "reg_lambda": {{"method": "suggest_float", "low": 0.0, "high": 2.0}},
                "max_delta_step": {{"method": "suggest_int", "low": 0, "high": 10}},
                "grow_policy": {{"method": "suggest_categorical", "choices": ["depthwise", "lossguide"]}}
            }}
        }}

        Only output valid JSON. No markdown, no explanations, no additional text.
        """
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model_name}. Supported models: catboost, xgboost"
        )

    logger.info("=== Prompt reveived by the API & to be sent to GPT-4.1 ===")
    logger.info(prompt)

    # Call Azure OpenAI chat endpoint
    response_text = call_azure_openai_chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in ML hyperparameter tuning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )

    logger.info("=== Response received from from GPT-4.1 ===")
    logger.info(response_text)

    cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.IGNORECASE)

    # Attempt to parse JSON response
    try:
        param_space = json.loads(cleaned)
        return {"model": model_name, "param_space": param_space}
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON returned by LLM:\n{response_text}"
        )

# ESG Analysis endpoint
@app.post("/esg_analysis")
async def generate_esg_analysis(request: ESGAnalysisRequest):
    """
    Generate comprehensive ESG analysis for Belgian real estate properties
    using Azure OpenAI with expert knowledge of Belgian regulations and market conditions.
    """
    
    # Extract property features for analysis
    property_data = request.propertyFeatures
    
    # Calculate detailed property characteristics for analysis
    epc_score = property_data.epcScore
    surface = property_data.habitableSurface
    year = property_data.buildingConstructionYear
    locality = property_data.locality
    province = property_data.province
    heating_type = property_data.heatingType
    
    # Calculate energy efficiency metrics
    is_old_building = year < 1980
    is_energy_efficient = epc_score in ['A_plus', 'A', 'B']
    needs_renovation = epc_score in ['E', 'F', 'G']
    
    # Calculate potential savings and renovations
    yearly_energy_cost = surface * (25 if needs_renovation else 15)
    potential_savings = yearly_energy_cost * (0.6 if needs_renovation else 0.3)
    renovation_cost = surface * (250 if needs_renovation else 100)
    
    # Build comprehensive ESG analysis prompt matching the original static analysis quality
    prompt = f"""
    You are a Belgian real estate energy efficiency and ESG expert. Generate a detailed analysis that matches the depth and specificity of professional Belgian property assessments.

    Property Details:
    - Location: {locality}, {province} ({property_data.postCode})
    - Type: {property_data.propertyType} - {property_data.subtype}
    - Surface: {surface}m²
    - Construction Year: {year}
    - EPC Score: {format_epc_score(epc_score)}
    - Heating: {heating_type}
    - Condition: {property_data.buildingCondition}
    - Estimated Price: {request.estimatedPrice:,.0f} €

    Generate exactly 6 detailed paragraphs following this structure:

    **Paragraph 1 - EPC Rating Analysis:**
    Start with "**EPC Rating Analysis:** With an EPC score of {format_epc_score(epc_score)} ({'among the best in Belgium' if is_energy_efficient else 'below current standards' if needs_renovation else 'good performance'}), this house is {'highly energy efficient and already exceeds current and near-future regulatory standards' if is_energy_efficient else 'flagged for potential renovation needs to meet upcoming 2030 energy standards' if needs_renovation else 'performing well but could benefit from targeted improvements'}."

    **Paragraph 2 - Energy Consumption Estimates:**
    Start with "**Energy Consumption Estimates:** For a {surface}m² house with an {format_epc_score(epc_score)} EPC, annual primary energy use typically ranges from {('180-300' if needs_renovation else '50-80' if is_energy_efficient else '100-150')} kWh/m²..." Include specific cost estimates based on the calculated yearly_energy_cost ({yearly_energy_cost}).

    **Paragraph 3 - Heating System Efficiency:**
    Start with "**Heating System Efficiency:** The {heating_type.lower().replace('_', ' ')} heating system is..." Provide detailed analysis of this specific heating type's efficiency, costs, and future viability in Belgium.

    **Paragraph 4 - Belgian Energy Performance Requirements:**
    Start with "**Belgian Energy Performance Requirements:** Flanders is tightening energy standards: by 2030, all homes must meet at least EPC label D for rentals..." Discuss how this property's {format_epc_score(epc_score)} rating relates to upcoming regulations.

    **Paragraph 5 - Rental Restrictions for Low-Performing Properties:**
    Start with "**Rental Restrictions for Low-Performing Properties:** Properties with EPC E or F {'will face rental bans and mandatory renovation requirements' if needs_renovation else 'are not an immediate concern for this property'}..." Discuss rental implications.

    **Paragraph 6 - Investment Recommendations:**
    Start with "**Investment Recommendations:** {'This property represents an excellent long-term investment with minimal energy upgrade risks. Focus on maintaining systems and consider smart home technologies for further optimization.' if is_energy_efficient else f'Priority renovations should target insulation, windows, and heating system upgrades. Estimated investment: €{renovation_cost:,.0f}, with annual savings of €{potential_savings:,.0f}.' if needs_renovation else 'Consider targeted efficiency improvements like smart thermostats, improved insulation, or renewable energy integration to enhance both comfort and future-proofing.'}"

    Each paragraph should be substantial (4-6 sentences) and include:
    - Specific Belgian regulations and standards
    - Concrete numbers and cost estimates
    - Regional context for {province}
    - Future regulatory changes and their impact
    - Actionable investment advice

    Make each paragraph detailed and informative, matching the depth of professional property assessments in Belgium.
    """

    # Call Azure OpenAI for ESG analysis
    response_text = call_azure_openai_chat(
        messages=[
            {"role": "system", "content": "You are a Belgian real estate ESG analysis expert with comprehensive knowledge of environmental, social, and governance factors in Belgian property markets."},
            {"role": "user", "content": prompt}
        ]
    )

    # Parse the response to extract structured ESG data
    # For now, return the full analysis text
    # In the future, could parse specific scores and recommendations
    
    # Parse key analysis points from the response
    analysis_points = []
    recommendations = []
    esg_scores = {"environmental": 7.0, "social": 7.0, "governance": 7.0, "overall": 7.0}
    
    # Extract key points and recommendations from the response
    lines = response_text.split('\n')
    current_section = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith('**') and line.endswith('**'):
            current_section = line.replace('**', '').lower()
        elif line.startswith('- ') and 'recommendation' in current_section:
            recommendations.append(line[2:])
        elif line.startswith('- ') or line.startswith('• '):
            analysis_points.append(line[2:] if line.startswith('- ') else line[2:])
        elif 'score:' in line.lower():
            # Try to extract numerical scores
            try:
                if 'environmental' in line.lower():
                    score = float(line.split(':')[1].split('/')[0].strip())
                    esg_scores["environmental"] = score
                elif 'social' in line.lower():
                    score = float(line.split(':')[1].split('/')[0].strip())
                    esg_scores["social"] = score
                elif 'governance' in line.lower():
                    score = float(line.split(':')[1].split('/')[0].strip())
                    esg_scores["governance"] = score
                elif 'overall' in line.lower():
                    score = float(line.split(':')[1].split('/')[0].strip())
                    esg_scores["overall"] = score
            except (ValueError, IndexError):
                pass  # Keep default scores if parsing fails
    
    # Calculate overall score if not found
    if esg_scores["overall"] == 7.0:
        esg_scores["overall"] = round((esg_scores["environmental"] + esg_scores["social"] + esg_scores["governance"]) / 3, 1)
    
    compliance_status = {
        "energy_compliance": "Compliant" if property_data.epcScore in ["A_plus", "A", "B"] else "Needs Review",
        "building_codes": "Compliant",
        "safety_standards": "Compliant" if property_data.buildingCondition in ["AS_NEW", "GOOD"] else "Needs Assessment"
    }
    
    financial_impact = {
        "energy_cost_annual": f"Estimated {500 + (ord(property_data.epcScore[0]) - ord('A')) * 200} €/year based on EPC {property_data.epcScore}",
        "improvement_cost_estimate": "5,000 - 25,000 € for energy efficiency upgrades",
        "roi_potential": f"ESG improvements could increase property value by {esg_scores['overall'] * 2:.0f}%"
    }
    
    return ESGAnalysisResponse(
        analysis_points=analysis_points[:10] if analysis_points else ["Comprehensive ESG analysis completed", "Property assessment performed"],
        esg_scores=esg_scores,
        recommendations=recommendations[:5] if recommendations else ["Review energy efficiency improvements", "Consider accessibility upgrades"],
        compliance_status=compliance_status,
        financial_impact=financial_impact,
        full_report=response_text
    )


# Quick ESG Analysis endpoint for consistent scoring
@app.post("/esg_quick_analysis")
async def generate_quick_esg_analysis(request: ESGAnalysisRequest):
    """
    Generate quick ESG assessment using the same AI model as detailed analysis
    but with condensed output for consistency.
    """
    
    # Extract property features
    property_data = request.propertyFeatures
    epc_score = property_data.epcScore
    surface = property_data.habitableSurface
    year = property_data.buildingConstructionYear
    heating_type = property_data.heatingType
    
    # Quick analysis prompt - focused on scoring consistency
    prompt = f"""
    You are a Belgian real estate ESG expert. Provide a quick ESG assessment that matches detailed analysis standards.

    Property: {property_data.propertyType} in {property_data.locality}, {property_data.province}
    EPC: {format_epc_score(epc_score)} | Year: {year} | Surface: {surface}m² | Heating: {heating_type}

    Generate consistent ESG scores on a 0-10 scale:

    ENVIRONMENT (consider EPC rating, construction year, heating system):
    Score: X.X/10
    Brief: One line assessment focusing on energy efficiency

    SOCIAL (consider accessibility, location, community impact):
    Score: X.X/10  
    Brief: One line assessment focusing on social benefits

    GOVERNANCE (consider building compliance, safety standards):
    Score: X.X/10
    Brief: One line assessment focusing on governance aspects

    OVERALL ESG RATING: 
    Score: X.X/10
    Grade: A+/A/B+/B/C/D (based on 8.5+=A+, 7.5-8.4=A, 6.5-7.4=B+, 5.5-6.4=B, 4.5-5.4=C, <4.5=D)

    Keep responses concise but substantive. Use same scoring criteria as detailed ESG analysis.
    """

    # Call Azure OpenAI with same parameters as detailed analysis
    response_text = call_azure_openai_chat(
        messages=[
            {"role": "system", "content": "You are a Belgian real estate ESG assessment expert. Provide consistent, reliable ESG scoring."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Same low temperature for consistency
        max_tokens=800
    )

    # Parse scores from the response
    environmental_score = 7.0
    social_score = 7.0  
    governance_score = 7.0
    overall_score = 7.0
    overall_grade = "B+"
    
    # Extract numerical scores using regex
    env_match = re.search(r'ENVIRONMENT.*?Score:\s*(\d+\.?\d*)/10', response_text, re.DOTALL | re.IGNORECASE)
    if env_match:
        environmental_score = float(env_match.group(1))
    
    social_match = re.search(r'SOCIAL.*?Score:\s*(\d+\.?\d*)/10', response_text, re.DOTALL | re.IGNORECASE)
    if social_match:
        social_score = float(social_match.group(1))
    
    gov_match = re.search(r'GOVERNANCE.*?Score:\s*(\d+\.?\d*)/10', response_text, re.DOTALL | re.IGNORECASE)
    if gov_match:
        governance_score = float(gov_match.group(1))
    
    overall_match = re.search(r'OVERALL.*?Score:\s*(\d+\.?\d*)/10', response_text, re.DOTALL | re.IGNORECASE)
    if overall_match:
        overall_score = float(overall_match.group(1))
    
    grade_match = re.search(r'Grade:\s*([A-D][+]?)', response_text, re.IGNORECASE)
    if grade_match:
        overall_grade = grade_match.group(1)
    
    # Calculate overall score if not found
    if overall_score == 7.0 and (environmental_score != 7.0 or social_score != 7.0 or governance_score != 7.0):
        overall_score = round((environmental_score + social_score + governance_score) / 3, 1)
    
    # Generate grade based on score
    if overall_score >= 8.5:
        overall_grade = "A+"
    elif overall_score >= 7.5:
        overall_grade = "A"
    elif overall_score >= 6.5:
        overall_grade = "B+"
    elif overall_score >= 5.5:
        overall_grade = "B"
    elif overall_score >= 4.5:
        overall_grade = "C"
    else:
        overall_grade = "D"

    # Return structured response
    esg_scores = {
        "environmental": environmental_score,
        "social": social_score,
        "governance": governance_score,
        "overall": overall_score
    }
    
    return {
        "esg_scores": esg_scores,
        "overall_grade": overall_grade,
        "analysis_summary": response_text,
        "confidence_level": "high"
    }
    prompt = f"""
    As a Belgian real estate ESG expert, provide a QUICK assessment for this property:
    
    Property: {property_data.propertyType} in {property_data.locality}, {property_data.province}
    Surface: {surface}m², Year: {year}, EPC: {format_epc_score(epc_score)}, Heating: {heating_type}
    
    Provide ONLY:
    1. Environmental score (0-10): Based on EPC rating and energy efficiency
    2. Social score (0-10): Based on location and family-friendliness  
    3. Governance score (0-10): Based on building age and condition
    4. Overall ESG score (0-10): Average of the three scores
    5. Three short insights (one per category)
    
    Format your response EXACTLY like this:
    Environmental Score: X.X/10
    Social Score: X.X/10  
    Governance Score: X.X/10
    Overall ESG Score: X.X/10
    
    Environmental Insight: [brief insight]
    Social Insight: [brief insight]
    Governance Insight: [brief insight]
    """
    
    try:
        # Call Azure OpenAI for quick analysis
        response_text = call_azure_openai_chat(
            messages=[
                {"role": "system", "content": "You are a Belgian real estate ESG expert. Provide consistent, professional assessments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=500
        )
        
        # Parse the response to extract scores and insights
        lines = response_text.strip().split('\n')
        esg_scores = {"environmental": 7.0, "social": 7.0, "governance": 7.0, "overall": 7.0}
        insights = {"environment": [], "social": [], "governance": []}
        
        for line in lines:
            line = line.strip()
            if "Environmental Score:" in line:
                try:
                    score = float(line.split('/')[0].split(':')[-1].strip())
                    esg_scores["environmental"] = score
                except:
                    pass
            elif "Social Score:" in line:
                try:
                    score = float(line.split('/')[0].split(':')[-1].strip())
                    esg_scores["social"] = score
                except:
                    pass
            elif "Governance Score:" in line:
                try:
                    score = float(line.split('/')[0].split(':')[-1].strip())
                    esg_scores["governance"] = score
                except:
                    pass
            elif "Overall ESG Score:" in line:
                try:
                    score = float(line.split('/')[0].split(':')[-1].strip())
                    esg_scores["overall"] = score
                except:
                    pass
            elif "Environmental Insight:" in line:
                insights["environment"].append(line.replace("Environmental Insight:", "").strip())
            elif "Social Insight:" in line:
                insights["social"].append(line.replace("Social Insight:", "").strip())
            elif "Governance Insight:" in line:
                insights["governance"].append(line.replace("Governance Insight:", "").strip())
        
        # Ensure overall score is calculated if not provided
        if esg_scores["overall"] == 7.0:
            esg_scores["overall"] = round((esg_scores["environmental"] + esg_scores["social"] + esg_scores["governance"]) / 3, 1)
        
        return {
            "esg_scores": esg_scores,
            "insights": insights,
            "analysis_type": "quick"
        }
        
    except Exception as e:
        print(f"Error in quick ESG analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate quick ESG analysis")


# Strategic summary endpoint
@app.post("/strategic_summary", response_model=StrategicSummaryResponse, tags=["ESG"])
async def generate_strategic_summary(request: StrategicSummaryRequest):
    """
    Generate a strategic summary for real estate properties
    focusing on ESG positioning and investment recommendations.
    """
    
    # Extract property features and ESG summary
    property_data = request.property_features
    esg_summary = request.esg_summary
    
    # Build strategic summary prompt
    prompt = f"""
    You are a Belgian real estate expert. Provide a strategic summary for a property based on its features and ESG summary.

    Property Features:
    - Type: {property_data.propertyType} - {property_data.subtype}
    - Location: {property_data.locality}, {property_data.province} ({property_data.postCode})
    - Surface: {property_data.habitableSurface}m²
    - Construction Year: {property_data.buildingConstructionYear}
    - EPC Score: {format_epc_score(property_data.epcScore)}
    - Heating: {property_data.heatingType}
    - Condition: {property_data.buildingCondition}

    ESG Summary:
    - Environmental Score: {esg_summary.environment}
    - Social Score: {esg_summary.social}
    - Governance Score: {esg_summary.governance}
    - Overall ESG Score: {esg_summary.overall}

    Strategic Goals: {request.strategic_goals}

    Provide a strategic positioning statement, ESG analysis, recommended actions, and clickable suggestions in JSON format.

    OUTPUT FORMAT:
    {{
        "strategic_positioning": "string",
        "esg_analysis": "string",
        "recommended_actions": "string",
        "clickable_suggestions": [{{"title": "string", "url": "string"}}],
        "confidence_score": float
    }}

    Only output valid JSON. No markdown, no explanations, no additional text.
    """
    
    logger.info("=== Strategic Summary Prompt ===")
    logger.info(prompt)
    
    # Call Azure OpenAI chat endpoint
    response_text = call_azure_openai_chat(
        messages=[
            {"role": "system", "content": "You are a Belgian real estate expert specializing in ESG analysis and investment strategies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800
    )

    logger.info("=== Strategic Summary Response ===")
    logger.info(response_text)
    
    # Attempt to parse JSON response
    try:
        summary_response = json.loads(response_text)
        return summary_response
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON returned by LLM:\n{response_text}"
        )

# Strategic Summary endpoint for unified ESG analysis
@app.post("/strategic-summary", response_model=StrategicSummaryResponse)
async def create_strategic_summary(request: StrategicSummaryRequest):
    """
    Generate unified strategic summary combining price prediction, ESG analysis, and agent insights
    for consistent user experience across all components.
    """
    
    property_data = request.property_features
    esg = request.esg_summary
    price = request.price_prediction
    
    # Build agent insights summary
    agent_summary = "\n".join([f"- {insight.agent}: {insight.summary}" for insight in request.agent_insights])
    
    # Create comprehensive prompt for strategic analysis
    prompt = f"""
    You are a Belgian real estate strategic advisor. Generate a comprehensive strategic summary that unifies all analysis components.

    PROPERTY OVERVIEW:
    - Location: {property_data.locality}, {property_data.province}
    - Type: {property_data.propertyType} - {property_data.subtype}
    - Predicted Value: €{price:,.0f}
    - EPC Rating: {format_epc_score(property_data.epcScore)}
    - Construction Year: {property_data.buildingConstructionYear}
    - Surface: {property_data.habitableSurface}m²
    - Strategic Goal: {request.strategic_goals}

    ESG PROFILE:
    - Environment Score: {esg.environment}/10
    - Social Score: {esg.social}/10  
    - Governance Score: {esg.governance}/10
    - Overall Rating: {esg.overall}

    AGENT INSIGHTS:
    {agent_summary}

    Generate exactly 4 sections:

    **STRATEGIC POSITIONING:**
    Analyze this property's position in the {property_data.locality} market. Consider the €{price:,.0f} price point, EPC {format_epc_score(property_data.epcScore)} rating, and {property_data.buildingConstructionYear} construction year. How does this compare to local market trends? Include specific market context for {property_data.province}.

    **ESG RISK & OPPORTUNITY ANALYSIS:**
    Based on the {esg.overall} ESG rating (E:{esg.environment}/10, S:{esg.social}/10, G:{esg.governance}/10), identify key risks and opportunities. Focus on Belgian energy regulations, upcoming EPC requirements, and investment implications. Address specific concerns for {format_epc_score(property_data.epcScore)} rated properties.

    **RECOMMENDED ACTIONS:**
    Provide 3-4 specific, actionable recommendations for a "{request.strategic_goals}" strategy. Include timeline, expected costs, and ROI projections. Consider Belgian tax incentives, renovation grants, and regulatory deadlines.

    **NEXT STEPS:**
    Suggest 3 clickable actions with titles and brief explanations that the user can take immediately.

    Keep each section substantial (3-4 sentences) but concise. Use European number formatting (spaces for thousands: €417 675).
    """

    # Call Azure OpenAI for strategic analysis
    response_text = call_azure_openai_chat(
        messages=[
            {"role": "system", "content": "You are a Belgian real estate strategic advisor with expertise in ESG analysis, market positioning, and investment strategies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for consistent analysis
        max_tokens=1200
    )

    # Parse the response into structured sections
    sections = response_text.split('**')
    strategic_positioning = ""
    esg_analysis = ""
    recommended_actions = ""
    
    for i, section in enumerate(sections):
        section = section.strip()
        if 'STRATEGIC POSITIONING' in section.upper():
            strategic_positioning = sections[i+1].strip() if i+1 < len(sections) else ""
        elif 'ESG RISK' in section.upper():
            esg_analysis = sections[i+1].strip() if i+1 < len(sections) else ""
        elif 'RECOMMENDED ACTIONS' in section.upper():
            recommended_actions = sections[i+1].strip() if i+1 < len(sections) else ""
    
    # Extract clickable suggestions (look for patterns like [Action Title])
    suggestions = []
    suggestion_patterns = re.findall(r'\[([^\]]+)\]', response_text)
    for i, suggestion in enumerate(suggestion_patterns[:3]):
        suggestions.append({
            "id": i + 1,
            "title": suggestion,
            "description": f"Click to explore {suggestion.lower()}",
            "action": suggestion.lower().replace(' ', '_')
        })
    
    # If no suggestions found, provide defaults
    if not suggestions:
        suggestions = [
            {"id": 1, "title": "Compare Market Properties", "description": "Analyze similar properties in the area", "action": "compare_market"},
            {"id": 2, "title": "ESG Improvement Calculator", "description": "Calculate ROI for energy upgrades", "action": "esg_calculator"},
            {"id": 3, "title": "Investment Strategy Guide", "description": "Get personalized investment recommendations", "action": "investment_guide"}
        ]
    
    # Calculate confidence score based on data completeness
    confidence = 0.8  # Base confidence
    if esg.environment > 0 and esg.social > 0 and esg.governance > 0:
        confidence += 0.1
    if len(request.agent_insights) > 2:
        confidence += 0.1
    
    return StrategicSummaryResponse(
        strategic_positioning=strategic_positioning or "Market analysis in progress...",
        esg_analysis=esg_analysis or "ESG assessment completed with current data.",
        recommended_actions=recommended_actions or "Strategic recommendations being prepared...",
        clickable_suggestions=suggestions,
        confidence_score=min(confidence, 1.0)
    )
app.include_router(router)