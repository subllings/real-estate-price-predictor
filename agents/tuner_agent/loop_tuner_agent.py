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

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/loop_tuner_agent.py <model_name>")
        print("Example: python scripts/loop_tuner_agent.py xgboost")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"Starting tuner loop for model: {model_name}")

    while True:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running tuning cycle...")
        orchestrator = TunerAgentOrchestrator(model_name)
        best_trial, is_perfect = orchestrator.run()  # <-- attention : run() doit retourner (best_trial, is_perfect)

        if is_perfect:
            print(f"🎯 Perfect model found (R² >= {PERFECT_R2_THRESHOLD}). Stopping tuning loop.")
            print("Best trial parameters:")
            for k, v in best_trial.params.items():
                print(f"  {k}: {v}")
            break

        if is_time_to_stop():
            print(f"⏰ Stop time reached ({STOP_HOUR:02d}:{STOP_MINUTE:02d}), stopping tuning loop.")
            break

        time.sleep(5)  # 5 seconds d’attente entre les runs pour ne pas spammer le CPU


def is_time_to_stop() -> bool:
    now = datetime.datetime.now()
    # Arrêt si on est passé à 7h00 ou plus
    return now.hour > STOP_HOUR or (now.hour == STOP_HOUR and now.minute >= STOP_MINUTE)

if __name__ == "__main__":
    main()
