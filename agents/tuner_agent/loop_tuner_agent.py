import sys, os
import time
import datetime
import argparse
import logging

# Add the project root to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configuration des logs Azure (doit être fait avant les imports Azure)
from utils.configure_logging import configure_azure_logging

from agents.tuner_agent.tuner_agent_orchestrator import TunerAgentOrchestrator

# === Thresholds to stop if a "perfect" model is found ===
PERFECT_R2_THRESHOLD = 0.95
PERFECT_MAE_THRESHOLD = 10000
PERFECT_RMSE_THRESHOLD = 15000


class TunerLoopRunner:
    def __init__(self, model_name: str, stop_hour: int = None, stop_minute: int = 0, 
                 no_time_limit: bool = False, duration_hours: float = None, 
                 end_time: str = None, max_trials: int = None):
        self.model_name = model_name
        self.stop_hour = stop_hour
        self.stop_minute = stop_minute
        self.no_time_limit = no_time_limit
        self.duration_hours = duration_hours
        self.end_time = end_time
        self.max_trials = max_trials
        self.start_time = datetime.datetime.now()

        self._validate_config()

    def _validate_config(self):
        termination_methods = sum([
            self.no_time_limit,
            self.stop_hour is not None,
            self.duration_hours is not None,
            self.end_time is not None,
            self.max_trials is not None
        ])
        
        if termination_methods == 0:
            print("⚠️ No termination condition specified. Use --no-time-limit, --stop-hour, --duration-hours, --end-time, or --max-trials.")
            sys.exit(1)
        elif termination_methods > 1:
            print("⚠️ Multiple termination conditions specified. Choose only one.")
            sys.exit(1)

    def run(self):
        print(f">>> Starting tuner loop for model: {self.model_name}")
        if self.no_time_limit:
            print("🕒 No time limit enabled: the loop will run indefinitely.")
        elif self.duration_hours:
            end_time = self.start_time + datetime.timedelta(hours=self.duration_hours)
            print(f"⏱️ Loop will run for {self.duration_hours} hours until {end_time.strftime('%H:%M:%S')}.")
        elif self.end_time:
            print(f"🛑 Loop will stop at {self.end_time} system time.")
        elif self.max_trials:
            print(f"🎯 Loop will stop after {self.max_trials} trials.")
        else:
            print(f"🛑 Loop will stop at {self.stop_hour:02d}:{self.stop_minute:02d} system time.")

        trial_count = 0
        while True:
            trial_count += 1
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running tuning cycle #{trial_count}...")
            
            # Adjust trials per cycle based on max_trials
            trials_this_cycle = 1 if self.max_trials and self.max_trials <= 10 else 50
            if self.max_trials and trial_count >= self.max_trials:
                trials_this_cycle = 1  # Last trial
            
            orchestrator = TunerAgentOrchestrator(self.model_name, n_trials=trials_this_cycle)
            best_trial, is_perfect = orchestrator.run()

            print(f"[DEBUG] is_perfect = {is_perfect}, trial #{trial_count}")
            if is_perfect:
                print(f"🎯 Perfect model found (R² >= {PERFECT_R2_THRESHOLD}). Stopping tuning loop.")
                print("Best trial parameters:")
                for k, v in best_trial.params.items():
                    print(f"  {k}: {v}")
                break

            # Check termination conditions
            if self.max_trials and trial_count >= self.max_trials:
                print(f"🎯 Maximum trials ({self.max_trials}) reached, stopping tuning loop.")
                break
            elif self.duration_hours and self.is_duration_exceeded():
                print(f"⏱️ Duration limit ({self.duration_hours} hours) reached, stopping tuning loop.")
                break
            elif self.end_time and self.is_end_time_reached():
                print(f"⏰ End time ({self.end_time}) reached, stopping tuning loop.")
                break
            elif not self.no_time_limit and self.stop_hour is not None and self.is_time_to_stop():
                print(f"⏰ Stop time reached ({self.stop_hour:02d}:{self.stop_minute:02d}), stopping tuning loop.")
                break

            print("[LOOP] Sleeping 5s before next tuning cycle...\n")
            time.sleep(5)

    def is_time_to_stop(self) -> bool:
        now = datetime.datetime.now()
        return now.hour > self.stop_hour or (now.hour == self.stop_hour and now.minute >= self.stop_minute)

    def is_duration_exceeded(self) -> bool:
        elapsed = datetime.datetime.now() - self.start_time
        return elapsed.total_seconds() / 3600 >= self.duration_hours

    def is_end_time_reached(self) -> bool:
        now = datetime.datetime.now()
        target_hour, target_minute = map(int, self.end_time.split(':'))
        return now.hour > target_hour or (now.hour == target_hour and now.minute >= target_minute)


def parse_args():
    parser = argparse.ArgumentParser(description="Tuner loop runner")
    parser.add_argument("model_name", type=str, help="Name of the model (xgboost, catboost, lightgbm, random_forest, stack_ensemble...)")
    
    # Termination options (mutually exclusive)
    termination_group = parser.add_mutually_exclusive_group()
    termination_group.add_argument("--no-time-limit", action="store_true", help="Run indefinitely (disables time-based stop)")
    termination_group.add_argument("--stop-hour", type=int, help="Hour (0-23) at which to stop the loop")
    termination_group.add_argument("--duration-hours", type=float, help="Duration in hours to run (e.g., 2.5 for 2h30m)")
    termination_group.add_argument("--end-time", type=str, help="End time in HH:MM format (e.g., 07:00)")
    termination_group.add_argument("--max-trials", type=int, help="Maximum number of trials to run")
    
    parser.add_argument("--stop-minute", type=int, default=0, help="Minute at which to stop the loop (used with --stop-hour)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loop_runner = TunerLoopRunner(
        model_name=args.model_name,
        stop_hour=args.stop_hour,
        stop_minute=args.stop_minute,
        no_time_limit=args.no_time_limit,
        duration_hours=args.duration_hours,
        end_time=args.end_time,
        max_trials=args.max_trials,
    )
    loop_runner.run()
