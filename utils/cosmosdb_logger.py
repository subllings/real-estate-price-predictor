import os
from datetime import datetime
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv
from utils.constants import ML_READY_DATA_FILE, TEST_MODE
import numpy as np
from datetime import datetime


load_dotenv()

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")
COSMOS_SERVERLESS = os.getenv("COSMOS_SERVERLESS", "false").lower() == "true"

class CosmosDbLogger:
    def __init__(self):
        self.client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        self.database = self._get_or_create_database(COSMOS_DATABASE_NAME)
        self.container = self._get_or_create_container(self.database, COSMOS_CONTAINER_NAME)

    def _get_or_create_database(self, name):
        try:
            return self.client.create_database_if_not_exists(id=name)
        except exceptions.CosmosHttpResponseError as e:
            print(f"[CosmosDB] Database error: {e}")
            raise

    def _get_or_create_container(self, database, name):
        try:
            if COSMOS_SERVERLESS:
                # Serverless account - no throughput parameter
                return database.create_container_if_not_exists(
                    id=name,
                    partition_key=PartitionKey(path="/run_id")
                )
            else:
                # Provisioned throughput account
                return database.create_container_if_not_exists(
                    id=name,
                    partition_key=PartitionKey(path="/run_id"),
                    offer_throughput=400
                )
        except exceptions.CosmosHttpResponseError as e:
            print(f"[CosmosDB] Container error: {e}")
            raise

    def log_metrics(self, metrics_dict):
        try:
            metrics_dict["id"] = metrics_dict.get("run_id", str(datetime.utcnow()))
            self.container.create_item(body=metrics_dict)
            print("[✔] Logged training metrics to Cosmos DB.")
        except Exception as e:
            print(f"[✘] Failed to log to Cosmos DB: {e}")

    def erase_metrics(self):
        print("[!] Deleting all existing items in CosmosDB container...")
        try:
            items = list(self.container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True
            ))
            for item in items:
                self.container.delete_item(item=item["id"], partition_key=item["run_id"])
            print("[✔] All items deleted from Cosmos DB.")
        except Exception as e:
            print(f"[✘] Failed to erase metrics: {e}")

    def get_best_run(self, model_name: str):
        query = """
        SELECT * FROM c
        WHERE c.model_name = @model_name AND c.agent_finetuning_ready = true
        ORDER BY c.r2_test DESC
        """
        parameters = [{"name": "@model_name", "value": model_name}]
        items = list(self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        return items[0] if items else None

    def delete_all_runs(self, model_name: str):
        query = f"SELECT * FROM c WHERE c.model_name = @model_name"
        parameters = [{"name": "@model_name", "value": model_name}]
        items = list(self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        for item in items:
            self.container.delete_item(item["id"], partition_key=item["model_name"])
        print(f"[✔] Deleted {len(items)} items for model: {model_name}")

    def get_best_run_hyperparams(self, model_name: str) -> dict:
        """
        Retrieve the hyperparameters of the best run (highest r2_test) for a given model.
        """
        best_run = self.get_best_run(model_name)
        if not best_run:
            print(f"[!] No runs found for model '{model_name}' with agent_finetuning_ready = true.")
            return {}

        # Extract hyperparameters - assumed to be under key 'hyperparameters'
        hyperparams = best_run.get("hyperparameters", {})
        print(f"[✔] Best hyperparameters retrieved for model '{model_name}': {hyperparams}")
        return hyperparams


    def log_best_trial(self, trial):

        log_data = {
            "id": f"best_trial_{datetime.utcnow().isoformat()}",
            "type": "best_trial",
            "timestamp": datetime.utcnow().isoformat(),
            "rmse": trial.value,
            "params": trial.params
        }
        self.container.upsert_item(log_data)



    def log_llm_response(self, source: str, model_name: str, payload: dict, response: str):
        try:
            log_data = {
                "id": f"llm_response_{datetime.utcnow().isoformat()}",
                "type": "llm_response",
                "timestamp": datetime.utcnow().isoformat(),
                "source": source,
                "model_name": model_name,
                "payload": self._convert_np_types(payload),
                "response": str(response)[:5000]  # tronque la réponse si trop longue
            }
            self.container.create_item(body=log_data)
            print("[✔] LLM response logged to Cosmos DB.")
        except Exception as e:
            print(f"[✘] Failed to log LLM response: {e}")


    def _convert_np_types(self, data):
        if isinstance(data, dict):
            return {k: self._convert_np_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_np_types(item) for item in data]
        elif isinstance(data, np.generic):
            return data.item()
        elif isinstance(data, bool):  # ajout nécessaire
            return bool(data)
        else:
            return data

    def log_experiment(self, data: dict):
        try:
            # Ajout timestamp et ID si absent
            if "run_id" not in data:
                data["run_id"] = f"exp_{datetime.utcnow().isoformat()}"
            if "id" not in data:
                data["id"] = data["run_id"]
            data["timestamp"] = datetime.utcnow().isoformat()

            # Calcul des deltas et interprétation
            train = data.get("metrics", {}).get("train", {})
            test = data.get("metrics", {}).get("test", {})

            if train and test:
                data["delta_mae"] = train.get("mae", 0) - test.get("mae", 0)
                data["delta_rmse"] = train.get("rmse", 0) - test.get("rmse", 0)
                data["delta_r2"] = train.get("r2", 0) - test.get("r2", 0)

                delta_r2 = data["delta_r2"]
                delta_rmse = data["delta_rmse"]

                if abs(delta_r2) < 0.02 and abs(delta_rmse) < 10000:
                    data["fit_status"] = "good_generalization"
                elif delta_r2 > 0.05 and delta_rmse > 20000:
                    data["fit_status"] = "severe_overfit"
                elif delta_r2 > 0.02:
                    data["fit_status"] = "slight_overfit"
                elif delta_r2 < -0.02:
                    data["fit_status"] = "underfit"
                else:
                    data["fit_status"] = "unclear"

                # Déduction de "is_perfect"
                data["is_perfect"] = (
                    data["fit_status"] == "good_generalization" and data["delta_r2"] >= 0 and test.get("r2", 0) >= 0.85)

                fit_comments = {
                    "good_generalization": "Model generalizes well to unseen data.",
                    "severe_overfit": "Warning: severe overfitting detected — test performance drops significantly.",
                    "slight_overfit": "Mild overfitting detected — consider more regularization or early stopping.",
                    "underfit": "Model may be underfitting — try increasing complexity or reducing regularization.",
                    "unclear": "Fit status unclear — delta metrics are inconclusive."
                }
                data["fit_comment"] = fit_comments.get(data["fit_status"], "No comment available.")



            # Conversion JSON-compatible
            clean_data = self._convert_np_types(data)

            # Écriture en base Cosmos DB
            self.container.create_item(body=clean_data)
            print("[✔] Experiment log successfully pushed to Cosmos DB.")

        except Exception as e:
            print(f"[✘] Failed to log experiment to Cosmos DB: {e}")



    def get_trials_for_model(self, model_name: str, limit: int = 10) -> list:
        """
        Retrieve the last 'limit' trials for a given model.
        """
        query = """
        SELECT TOP @limit * FROM c
        WHERE c.model_name = @model_name AND c.type = 'metrics'
        ORDER BY c.timestamp DESC
        """
        parameters = [
            {"name": "@limit", "value": limit},
            {"name": "@model_name", "value": model_name}
        ]
        try:
            return list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
        except Exception as e:
            print(f"[✘] Error fetching trials from CosmosDB: {e}")
            return []
