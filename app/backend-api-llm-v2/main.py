import os
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import json
import logging

logger = logging.getLogger("uvicorn.error")

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
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
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
    predictionTop: Optional[float] = None  # Make it optional
    userProfile: UserProfile

class LLMParamRequest(BaseModel):
    model_name: str

class SuggestionRequest(BaseModel):
    model_name: str
    previous_trials: list

class ESGUserProfile(BaseModel):
    name: str
    type: str
    objectives: List[str]
    language: str

class ESGAnalysisRequest(BaseModel):
    propertyType: str
    subtype: Optional[str] = None
    province: str
    locality: str
    postCode: Optional[str] = None
    constructionYear: Optional[int] = None
    surface: Optional[float] = None
    condition: Optional[str] = None
    epcScore: Optional[str] = None
    heatingType: Optional[str] = None
    estimatedPrice: float
    userProfile: ESGUserProfile

class ESGAnalysisResponse(BaseModel):
    comments: List[str]

# === Utility Function ===

def call_azure_openai_chat(messages: List[dict], temperature: float = 0.7, max_tokens: int = 300):
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

    # Build prediction information based on available models
    prediction_info = f"Model (all features) predicted: {request.predictionAll} EUR"
    if request.predictionTop is not None:
        prediction_info += f"\nModel (top 30 features) predicted: {request.predictionTop} EUR"
    
    prompt = f"""
    You are a real estate data assistant for a user profile of type '{profile.type}', 
    with objectives: {', '.join(profile.objectives)}. 
    The property is located in {form.locality}, {form.province}, {form.region}.
    
    {prediction_info}
    Model scores: MAE = {form.scoreMeta.mae}, RMSE = {form.scoreMeta.rmse}, R² = {form.scoreMeta.r2}

    Based on this, generate 2-3 smart, business-oriented comments tailored to the user's profile.
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

@router.post("/esg-analysis", tags=["ESG Analysis"], response_model=ESGAnalysisResponse)
def perform_esg_analysis(request: ESGAnalysisRequest):
    """
    Provides intelligent ESG analysis for real estate properties.
    Analyzes energy efficiency, sustainability, and 2030 compliance.
    """
    
    # Build property context
    property_context = f"""
    Property Type: {request.propertyType}
    Location: {request.locality}, {request.province}
    Construction Year: {request.constructionYear or 'Unknown'}
    Surface: {request.surface or 'Unknown'} m²
    Condition: {request.condition or 'Unknown'}
    EPC Score: {request.epcScore or 'Unknown'}
    Heating Type: {request.heatingType or 'Unknown'}
    Estimated Price: €{request.estimatedPrice:,.0f}
    """

    # Create comprehensive ESG analysis prompt
    prompt = f"""
    You are an expert ESG (Environmental, Social, Governance) analyst specializing in Belgian real estate.
    
    Analyze this property for energy efficiency, sustainability, and 2030 compliance:
    
    {property_context}
    
    Provide a comprehensive ESG analysis focusing on:
    
    1. **Energy Performance Assessment**
       - Current EPC rating analysis and implications
       - Energy consumption estimates
       - Heating system efficiency evaluation
    
    2. **2030 Compliance & Regulations**
       - Belgian energy performance requirements
       - Rental restrictions for low-performing properties
       - Timeline for mandatory improvements
    
    3. **Renovation Recommendations**
       - Priority improvements for energy efficiency
       - Estimated costs and ROI
       - Available grants and subsidies in {request.province}
    
    4. **Market Impact Analysis**
       - Effect of energy performance on property value
       - Future marketability considerations
       - Green premium potential
    
    5. **Financial Projections**
       - Energy cost savings potential
       - Renovation investment requirements
       - Long-term value preservation
    
    Provide practical, actionable insights in a conversational tone. Focus on specific recommendations for this property type and location.
    
    Structure your response as distinct analysis points, each 2-3 sentences long.
    """

    try:
        # Call Azure OpenAI for ESG analysis
        response_text = call_azure_openai_chat(
            messages=[
                {"role": "system", "content": "You are an expert ESG analyst for Belgian real estate, providing practical energy efficiency and sustainability advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1000   # Longer response for comprehensive analysis
        )
        
        logger.info(f"ESG Analysis completed for {request.propertyType} in {request.locality}")
        
        # Split response into logical analysis points
        analysis_points = []
        
        # Split by paragraphs and clean up
        paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            # Further split long paragraphs by sentences if needed
            sentences = paragraph.split('. ')
            if len(sentences) > 3:
                # Group sentences into chunks of 2-3
                for i in range(0, len(sentences), 3):
                    chunk = '. '.join(sentences[i:i+3])
                    if chunk and len(chunk) > 50:  # Minimum meaningful length
                        analysis_points.append(chunk.rstrip('.') + '.')
            else:
                if len(paragraph) > 50:  # Minimum meaningful length
                    analysis_points.append(paragraph)
        
        # Ensure we have meaningful analysis points
        if not analysis_points:
            analysis_points = [
                f"Energy Performance: Property in {request.locality} shows potential for efficiency improvements.",
                f"2030 Compliance: Based on {request.constructionYear or 'age'}, renovation planning recommended.",
                f"Market Value: Energy upgrades can enhance property value in {request.province} market."
            ]
        
        return ESGAnalysisResponse(comments=analysis_points)
        
    except Exception as e:
        logger.error(f"ESG Analysis failed: {str(e)}")
        
        # Provide fallback analysis based on available data
        fallback_comments = [
            f"Property Assessment: {request.propertyType} in {request.locality} requires detailed energy evaluation.",
            f"Regulatory Context: Belgian 2030 energy standards impact properties built before 2010.",
            f"Investment Opportunity: Energy efficiency improvements can enhance both comfort and value.",
            f"Next Steps: Professional EPC assessment recommended for {request.surface or 'this'} m² property."
        ]
        
        return ESGAnalysisResponse(comments=fallback_comments)


app.include_router(router)