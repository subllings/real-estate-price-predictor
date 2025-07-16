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
    predictionTop: float
    userProfile: UserProfile

class LLMParamRequest(BaseModel):
    model_name: str

class SuggestionRequest(BaseModel):
    model_name: str
    previous_trials: list

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


app.include_router(router)