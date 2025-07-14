import sys
import os
import time

# Set environment variables to limit thread usage for some libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from agents.tuner_agent.tuner_agent_orchestrator import TunerAgentOrchestrator

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_tuner_agent.py <model_name>")
        print("Example: python scripts/run_tuner_agent.py xgboost")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"[run_tuner_agent.py] Launching tuning agent for model: {model_name}")

    orchestrator = TunerAgentOrchestrator(model_name)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Tuning iteration #{iteration} ---")
        
        # Run the tuning process and get the best trial and whether the model is perfect
        best_trial, is_perfect = orchestrator.run()

        print(f"Is perfect? {is_perfect}")
        if is_perfect:
            print("Perfect model found! Stopping loop early.")
            print("Best trial parameters:")
            for k, v in best_trial.params.items():
                print(f"  {k}: {v}")
            break  # Exit the loop if perfect model is found

        # Optional: wait a bit before next iteration to avoid overload
        time.sleep(5)

if __name__ == "__main__":
    main()
