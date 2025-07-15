// API Configuration for Development and Production

const isDevelopment = process.env.NODE_ENV === 'development' || !process.env.NODE_ENV;

// Development URLs (local)
const DEV_CONFIG = {
  PREDICTION_API_URL: "http://127.0.0.1:8000",
  LLM_API_URL: "http://127.0.0.1:8010",
  CHAT_API_URL: "http://127.0.0.1:8010/chat",
  COMMENT_API_URL: "http://127.0.0.1:8010/comment",
  ESG_ANALYSIS_API_URL: "http://127.0.0.1:8010/esg-analysis"
};

// Production URLs (Azure)
const PROD_CONFIG = {
  PREDICTION_API_URL: "https://realestate-api.azurewebsites.net",
  LLM_API_URL: "https://realestate-api-llm-v2.azurewebsites.net",
  CHAT_API_URL: "https://realestate-api-llm-v2.azurewebsites.net/chat",
  COMMENT_API_URL: "https://realestate-api-llm-v2.azurewebsites.net/comment",
  ESG_ANALYSIS_API_URL: "https://realestate-api-llm-v2.azurewebsites.net/esg-analysis"
};

// Export the appropriate configuration based on environment
const API_CONFIG = isDevelopment ? DEV_CONFIG : PROD_CONFIG;

export const {
  PREDICTION_API_URL,
  LLM_API_URL,
  CHAT_API_URL,
  COMMENT_API_URL,
  ESG_ANALYSIS_API_URL
} = API_CONFIG;

export default API_CONFIG;
