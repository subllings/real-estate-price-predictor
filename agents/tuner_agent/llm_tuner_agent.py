import json
import requests
from utils.cosmosdb_logger import CosmosDbLogger
from utils.constants import TEST_MODE


class LLMTunerAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = CosmosDbLogger()
        self.api_url = "http://127.0.0.1:8010/llm-tuner/suggest-space"

    def run(self):
        print(f"[INFO] Running LLM tuning for model: {self.model_name}")
        best_hyperparams = self.get_best_run_hyperparams()
        if not best_hyperparams:
            print(f"[!] No valid hyperparameters found for model: {self.model_name}")
            return

        suggested_params = self.suggest_param_space()
        print(f"[INFO] Suggested parameter space for model {self.model_name}: {suggested_params}")

    

    def suggest_param_space(self) -> dict:
        """
        Call the internal FastAPI endpoint to suggest parameter space.
        """
        payload = {
            "model_name": self.model_name
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[❌] Failed to call internal API: {e}")

        raw = response.text

        print("\n" + "=" * 60)
        print(">>> API LLM Response:")
        print("=" * 60)
        print(raw)
        print("=" * 60 + "\n")

        # Log in CosmosDB
        self.logger.log_llm_response(
            source="LLMTunerAgent",
            model_name=self.model_name,
            payload=payload,
            response=raw
        )

        try:
            result = json.loads(raw)
            if "param_space" not in result:
                raise ValueError("Missing 'param_space' in LLM response")
            return result["param_space"]
        except json.JSONDecodeError:
            raise ValueError(f"[❌] LLM returned invalid JSON: {raw}")









    
    
    def get_best_run_hyperparams(self) -> dict:
        """
        Retrieve the hyperparameters of the best run (highest r2_test) for this model.
        """
        best_run = self.logger.get_best_run(model_name=self.model_name)
        if not best_run:
            print(f"[!] No runs found for model '{self.model_name}' with agent_finetuning_ready = true.")
            return {}

        hyperparams = best_run.get("hyperparameters", {})
        print(f"[✔] Best hyperparameters retrieved for model '{self.model_name}': {hyperparams}")
        return hyperparams
