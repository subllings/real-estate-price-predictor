import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ajoute la racine du projet au PYTHONPATH
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
    orchestrator.run()
    

if __name__ == "__main__":
    main()
