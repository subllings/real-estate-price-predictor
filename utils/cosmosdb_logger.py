import os
from datetime import datetime
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

load_dotenv()

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")

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
