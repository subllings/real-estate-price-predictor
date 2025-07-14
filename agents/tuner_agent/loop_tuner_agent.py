import sys, os
import time
import datetime

# Ajoute la racine du projet au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from agents.tuner_agent.tuner_agent_orchestrator import TunerAgentOrchestrator

# === Seuils pour arrêter si un "modèle parfait" est trouvé ===
PERFECT_R2_THRESHOLD = 0.95
PERFECT_MAE_THRESHOLD = 10000
PERFECT_RMSE_THRESHOLD = 15000

# === Heure limite d'arrêt ===
STOP_HOUR = 7
STOP_MINUTE = 0

def is_perfect_model(metrics: dict) -> bool:
    return (
        metrics.get("r2_test", 0) >= PERFECT_R2_THRESHOLD and
        metrics.get("mae_test", float("inf")) <= PERFECT_MAE_THRESHOLD and
        metrics.get("rmse_test", float("inf")) <= PERFECT_RMSE_THRESHOLD
    )

def is_time_to_stop() -> bool:
    now = datetime.datetime.now()
    return now.hour >= STOP_HOUR and now.minute >= STOP_MINUTE

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/loop_tuner_agent.py <model_name>")
        print("Example: python scripts/loop_tuner_agent.py xgboost")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"🚀 Starting tuner loop for model: {model_name}")

    while True:
        print("🔁 Running tuning cycle...")
        orchestrator = TunerAgentOrchestrator(model_name)
        best_metrics = orchestrator.run()  # Assure-toi q_
