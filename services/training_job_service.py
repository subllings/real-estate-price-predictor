import time
import threading
from utils.cosmosdb_logger import CosmosDbLogger

class TrainingJobService:
    def __init__(self):
        self.cosmos_logger = CosmosDbLogger()
        self.running = True
        self.active_jobs = {}
    
    def start(self):
        print("🚀 Training Job Service started - listening for remote commands...")
        while self.running:
            try:
                # Check for new jobs to start
                pending_jobs = self.cosmos_logger.get_training_jobs(status_filter="queued")
                
                for job in pending_jobs:
                    if job["id"] not in self.active_jobs:
                        self.start_training_job(job)
                
                # Check for stop requests
                self.check_stop_requests()
                
            except Exception as e:
                print(f"❌ Service error: {e}")
            
            time.sleep(10)  # Poll every 10 seconds
    
    def start_training_job(self, job_config):
        # Start training in separate thread
        thread = threading.Thread(target=self.run_training, args=(job_config,))
        thread.start()
        self.active_jobs[job_config["id"]] = thread
