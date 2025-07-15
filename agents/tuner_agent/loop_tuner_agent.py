import sys, os
import time
import datetime
import argparse

# Add the project root to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from agents.tuner_agent.tuner_agent_orchestrator import TunerAgentOrchestrator

# === Thresholds to stop if a "perfect" model is found ===
PERFECT_R2_THRESHOLD = 0.95
PERFECT_MAE_THRESHOLD = 10000
PERFECT_RMSE_THRESHOLD = 15000


class TunerLoopRunner:
    def __init__(self, model_name: str, stop_hour: int = None, stop_minute: int = 0, no_time_limit: bool = False):
        self.model_name = model_name
        self.stop_hour = stop_hour
        self.stop_minute = stop_minute
        self.no_time_limit = no_time_limit

        self._validate_config()

    def _validate_config(self):
        if not self.no_time_limit and self.stop_hour is None:
            print("⚠️ No stop time defined and no --no-time-limit provided. Exiting.")
            sys.exit(1)

    def run(self):
        print(f">>> Starting tuner loop for model: {self.model_name}")
        if self.no_time_limit:
            print("🕒 No time limit enabled: the loop will run indefinitely.")
        else:
            print(f"🛑 Loop will stop at {self.stop_hour:02d}:{self.stop_minute:02d} system time.")

        while True:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running tuning cycle...")
            orchestrator = TunerAgentOrchestrator(self.model_name)
            best_trial, is_perfect = orchestrator.run()

            print(f"[DEBUG] is_perfect = {is_perfect}")
            if is_perfect:
                print(f"🎯 Perfect model found (R² >= {PERFECT_R2_THRESHOLD}). Stopping tuning loop.")
                print("Best trial parameters:")
                for k, v in best_trial.params.items():
                    print(f"  {k}: {v}")
                break

            if not self.no_time_limit and self.is_time_to_stop():
                print(f"⏰ Stop time reached ({self.stop_hour:02d}:{self.stop_minute:02d}), stopping tuning loop.")
                break

            print("[LOOP] Sleeping 5s before next tuning cycle...\n")
            time.sleep(5)

    def is_time_to_stop(self) -> bool:
        now = datetime.datetime.now()
        return now.hour > self.stop_hour or (now.hour == self.stop_hour and now.minute >= self.stop_minute)


def parse_args():
    parser = argparse.ArgumentParser(description="Tuner loop runner")
    parser.add_argument("model_name", type=str, help="Name of the model (xgboost, catboost...)")
    parser.add_argument("--stop-hour", type=int, default=None, help="Hour (0-23) at which to stop the loop")
    parser.add_argument("--stop-minute", type=int, default=0, help="Minute at which to stop the loop")
    parser.add_argument("--no-time-limit", action="store_true", help="Run indefinitely (disables time-based stop)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loop_runner = TunerLoopRunner(
        model_name=args.model_name,
        stop_hour=args.stop_hour,
        stop_minute=args.stop_minute,
        no_time_limit=args.no_time_limit,
    )
    loop_runner.run()
