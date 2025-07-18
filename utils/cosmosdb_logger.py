import os
import logging
from datetime import datetime
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv
from utils.constants import ML_READY_DATA_FILE, TEST_MODE
import numpy as np
from datetime import datetime

# Configuration des logs Azure (doit être fait avant les imports Azure)
from utils.configure_logging import configure_azure_logging


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


    def log_trial(self, trial_info: dict):
        """
        Log trial information from hyperparameter tuning with enriched metrics.
        Compatible with both CatBoost and XGBoost tuners.
        """
        try:
            # Générer un ID unique et run_id si absent
            timestamp = datetime.utcnow().isoformat()
            trial_number = trial_info.get("trial_number", "unknown")
            model_type = trial_info.get("model_type", "unknown")
            
            if "run_id" not in trial_info:
                trial_info["run_id"] = f"{model_type}_trial_{trial_number}_{timestamp}"
            if "id" not in trial_info:
                trial_info["id"] = trial_info["run_id"]
            
            # Ajouter métadonnées standards
            trial_info.update({
                "type": "optuna_trial",
                "timestamp": timestamp,
                "source": "TunerAgent",
                "model_name": model_type
            })
            
            # Conversion JSON-compatible pour toutes les nouvelles métriques
            clean_data = self._convert_np_types(trial_info)
            
            # Écriture en base Cosmos DB
            self.container.create_item(body=clean_data)
            print(f"[✔] Trial {trial_number} ({model_type}) logged to Cosmos DB with enriched metrics.")
            
        except Exception as e:
            print(f"[✘] Failed to log trial to Cosmos DB: {e}")

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



    def get_trials_for_model(self, model_name: str, limit: int = 10, container_name: str = None) -> list:
        """
        Retrieve the last 'limit' trials based on 'model_name' with all enriched metrics.
        Can optionally specify a different container (like 'ModelMetrics' for structured metrics).
        """
        # Utiliser le conteneur spécifié ou le conteneur par défaut
        container = self.container
        if container_name:
            try:
                container = self.database.get_container_client(container_name)
            except Exception as e:
                print(f"[⚠] Could not access container '{container_name}', falling back to default: {e}")
                container = self.container
        
        query = """
        SELECT TOP @limit * FROM c
        WHERE c.model_name = @model_name AND c.type = 'optuna_trial'
        ORDER BY c.timestamp DESC
        """
        parameters = [
            {"name": "@limit", "value": limit},
            {"name": "@model_name", "value": model_name}
        ]
        try:
            trials = list(container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            container_info = f" from container '{container_name}'" if container_name else ""
            print(f"[✔] Retrieved {len(trials)} trials for model '{model_name}' with enriched metrics{container_info}.")
            return trials
        except Exception as e:
            print(f"[✘] Error fetching trials from CosmosDB: {e}")
            return []

    def get_trial_performance_analytics(self, model_name: str, limit: int = 50) -> dict:
        """
        Get performance analytics for trials including training times, iterations, etc.
        """
        trials = self.get_trials_for_model(model_name, limit)
        
        if not trials:
            return {}
        
        # Extraire les métriques pour analyse
        training_times = [t.get("training_time_seconds", 0) for t in trials if t.get("training_time_seconds")]
        trial_durations = [t.get("trial_duration_seconds", 0) for t in trials if t.get("trial_duration_seconds")]
        best_iterations = [t.get("best_iteration", 0) for t in trials if t.get("best_iteration")]
        rmse_scores = [t.get("mean_rmse", float('inf')) for t in trials if t.get("mean_rmse")]
        
        analytics = {
            "model_name": model_name,
            "total_trials": len(trials),
            "avg_training_time": np.mean(training_times) if training_times else 0,
            "avg_trial_duration": np.mean(trial_durations) if trial_durations else 0,
            "avg_best_iteration": np.mean(best_iterations) if best_iterations else 0,
            "best_rmse": min(rmse_scores) if rmse_scores else float('inf'),
            "avg_rmse": np.mean(rmse_scores) if rmse_scores else float('inf'),
            "feature_selection_methods": list(set([t.get("feature_selection_method", "unknown") for t in trials])),
            "cv_strategies": list(set([t.get("cv_strategy", "unknown") for t in trials])),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"[✔] Generated performance analytics for {model_name}: {len(trials)} trials analyzed.")
        return analytics


    def get_distinct_model_names(self, source="LLMTunerAgent") -> list:
        """
        Retrieve all distinct model names from the database, filtered by source.
        """
        query = {
            "query": f"SELECT DISTINCT c.model_name FROM c WHERE c.source = @source",
            "parameters": [{"name": "@source", "value": source}]
        }
        results = self.container.query_items(query=query, enable_cross_partition_query=True)
        return [item["model_name"] for item in results]

    def create_model_metrics_container(self, container_name: str = "ModelMetrics"):
        """
        Créer le container ModelMetrics pour les métriques structurées
        """
        try:
            if COSMOS_SERVERLESS:
                # Serverless account - no throughput parameter
                model_metrics_container = self.database.create_container_if_not_exists(
                    id=container_name,
                    partition_key=PartitionKey(path="/model_type")
                )
            else:
                # Provisioned throughput account
                model_metrics_container = self.database.create_container_if_not_exists(
                    id=container_name,
                    partition_key=PartitionKey(path="/model_type"),
                    offer_throughput=400
                )
            
            print(f"[✔] Container '{container_name}' créé/vérifié avec succès.")
            return model_metrics_container
            
        except exceptions.CosmosHttpResponseError as e:
            print(f"[CosmosDB] Erreur lors de la création du container {container_name}: {e}")
            raise

    def log_model_metrics(self, metrics: dict, container_name: str = "ModelMetrics"):
        """
        Logger les métriques de modèle dans le container ModelMetrics
        """
        try:
            # Créer/obtenir le container ModelMetrics
            model_metrics_container = self.create_model_metrics_container(container_name)
            
            # Générer un ID unique
            timestamp = datetime.utcnow().isoformat()
            metrics_id = f"model_metrics_{timestamp}"
            
            # Structure standardisée pour les métriques
            record = {
                "id": metrics_id,
                "model_type": metrics.get("model_type", "catboost"),
                "model_name": metrics.get("model_name", "CatBoost CV (All Features)"),
                "timestamp": timestamp,
                "trial_number": metrics.get("trial_number", 0),
                "experiment_name": metrics.get("experiment_name", ""),
                
                # Métriques de performance
                "r2_train": metrics.get("r2_train", 0.0),
                "r2_test": metrics.get("r2_test", 0.0),
                "mae_train": metrics.get("mae_train", 0.0),
                "mae_test": metrics.get("mae_test", 0.0),
                "rmse_train": metrics.get("rmse_train", 0.0),
                "rmse_test": metrics.get("rmse_test", 0.0),
                
                # Analyse de généralisation
                "r2_gap": metrics.get("r2_gap", 0.0),
                "generalization_status": metrics.get("generalization_status", "Unknown"),
                
                # Métadonnées du modèle
                "hyperparameters": metrics.get("hyperparameters", {}),
                "feature_importance": metrics.get("feature_importance", []),
                "training_time": metrics.get("training_time", 0.0),
                "n_features": metrics.get("n_features", 0),
                
                # Statut
                "status": metrics.get("status", "completed"),
                "is_production_ready": metrics.get("is_production_ready", False),
                
                # Métadonnées système
                "source": "catboost_tuner"
            }
            
            # Conversion JSON-compatible
            clean_record = self._convert_np_types(record)
            
            # Insérer dans le container ModelMetrics
            model_metrics_container.create_item(body=clean_record)
            
            print(f"[✔] Métriques de modèle loggées dans {container_name}: {metrics_id}")
            return metrics_id
            
        except Exception as e:
            print(f"[✘] Erreur lors du logging des métriques de modèle: {e}")
            raise

    def get_model_metrics(self, model_type: str = "catboost", limit: int = 100, container_name: str = "ModelMetrics"):
        """
        Récupérer les métriques de modèle depuis le container ModelMetrics
        """
        try:
            # Obtenir le container ModelMetrics
            model_metrics_container = self.database.get_container_client(container_name)
            
            query = """
                SELECT * FROM c 
                WHERE c.model_type = @model_type 
                ORDER BY c.timestamp DESC
                OFFSET 0 LIMIT @limit
            """
            
            parameters = [
                {"name": "@model_type", "value": model_type},
                {"name": "@limit", "value": limit}
            ]
            
            items = list(model_metrics_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            print(f"[✔] Récupéré {len(items)} métriques de modèle pour {model_type}")
            return items
            
        except Exception as e:
            print(f"[✘] Erreur lors de la récupération des métriques de modèle: {e}")
            return []

    def get_model_summary(self, model_type: str = "catboost", container_name: str = "ModelMetrics"):
        """
        Récupérer un résumé des métriques pour un type de modèle
        """
        try:
            metrics = self.get_model_metrics(model_type, limit=1000, container_name=container_name)
            
            if not metrics:
                return {
                    "total_experiments": 0,
                    "best_r2_score": 0,
                    "average_r2_score": 0,
                    "latest_experiment": None
                }
            
            # Calculer les statistiques
            r2_scores = [m.get("r2_test", 0) for m in metrics if m.get("r2_test", 0) > 0]
            
            # Trouver la meilleure expérience
            best_experiment = max(metrics, key=lambda x: x.get("r2_test", 0))
            
            # Trouver la dernière expérience
            latest_experiment = max(metrics, key=lambda x: x.get("timestamp", ""))
            
            summary = {
                "total_experiments": len(metrics),
                "best_r2_score": max(r2_scores) if r2_scores else 0,
                "average_r2_score": sum(r2_scores) / len(r2_scores) if r2_scores else 0,
                "latest_experiment": {
                    "id": latest_experiment.get("id", ""),
                    "model_type": latest_experiment.get("model_name", ""),
                    "r2_score": latest_experiment.get("r2_test", 0),
                    "timestamp": latest_experiment.get("timestamp", "")
                }
            }
            
            print(f"[✔] Résumé généré pour {model_type}: {len(metrics)} expériences")
            return summary
            
        except Exception as e:
            print(f"[✘] Erreur lors de la génération du résumé: {e}")
            return {
                "total_experiments": 0,
                "best_r2_score": 0,
                "average_r2_score": 0,
                "latest_experiment": None
            }