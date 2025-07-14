import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import json

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

    prompt = f"""
    You are a real estate data assistant for a user profile of type '{profile.type}', 
    with objectives: {', '.join(profile.objectives)}. 
    The property is located in {form.locality}, {form.province}, {form.region}.
    
    Model 1 (all features) predicted: {request.predictionAll} EUR  
    Model 2 (top 30 features) predicted: {request.predictionTop} EUR  
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


@app.post("/llm-tuner/suggest-space", tags=["LLM Tuner"])
def suggest_param_space(request: SuggestionRequest):
    model_name = request.model_name
    trials = request.previous_trials

    if not trials:
        trial_summary = "(no prior trials found)"
    else:
        trial_summary = "\n".join([
            f"- Params: {t.get('hyperparameters', t.get('params'))}, "
            f"Score: {t.get('r2_test') or t.get('rmse') or '?'}"
            for t in trials
        ])

    prompt = f"""
You are an expert in hyperparameter tuning using Optuna.
Here are previous trials for the model '{model_name}':

{trial_summary}

Based on this history, suggest a refined Optuna parameter space in JSON format.
Only output valid JSON. No explanations.
"""

    response_text = call_azure_openai_chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in ML hyperparameter tuning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )

    try:
        param_space = json.loads(response_text)
        return {"model": model_name, "param_space": param_space}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON returned by LLM:\n{response_text}")
