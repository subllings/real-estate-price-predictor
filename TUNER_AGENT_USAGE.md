# Tuner Agent Usage Guide

## Overview
The tuner agent has been updated to support the new training termination options from the React interface. It now supports multiple ways to control when training stops and includes support for all model types.

## Supported Models
- `catboost` - CatBoost + Optuna
- `xgboost` - XGBoost + Optuna  
- `lightgbm` - LightGBM + Optuna (fallback to XGBoost tuner)
- `random_forest` - Random Forest + Optuna (fallback to CatBoost tuner)
- `stack_ensemble` - Stacked Ensemble + Optuna (uses CatBoost base)

## Termination Options

### 1. No Time Limit (Endless Loop)
Run indefinitely until manually stopped or perfect model found:
```bash
./scripts/run_loop_tuner_agent.sh catboost --no-time-limit
```

### 2. Duration-Based Termination
Stop after a specific duration in hours:
```bash
# Run for 2.5 hours
./scripts/run_loop_tuner_agent.sh xgboost --duration-hours 2.5

# Run for 8 hours (overnight)
./scripts/run_loop_tuner_agent.sh lightgbm --duration-hours 8
```

### 3. End Time Termination
Stop at a specific time (24-hour format):
```bash
# Stop at 7:00 AM
./scripts/run_loop_tuner_agent.sh catboost --end-time 07:00

# Stop at 6:30 PM
./scripts/run_loop_tuner_agent.sh random_forest --end-time 18:30
```

### 4. Max Trials Termination
Stop after a specific number of trials:
```bash
# Run exactly 100 trials
./scripts/run_loop_tuner_agent.sh xgboost --max-trials 100

# Run 50 trials
./scripts/run_loop_tuner_agent.sh stack_ensemble --max-trials 50
```

### 5. Legacy Hour-Based Termination
Stop at specific hour and minute (legacy support):
```bash
# Stop at 6:30 AM
./scripts/run_loop_tuner_agent.sh catboost --stop-hour 6 --stop-minute 30
```

## Integration with React Interface

The tuner agent parameters now map directly to the React interface options:

| React Option | Script Parameter | Example |
|--------------|------------------|---------|
| Endless Loop | `--no-time-limit` | `--no-time-limit` |
| Time Duration | `--duration-hours` | `--duration-hours 2.5` |
| End Time | `--end-time` | `--end-time 07:00` |
| Max Trials | `--max-trials` | `--max-trials 100` |

## Perfect Model Detection
Training automatically stops if a "perfect" model is found:
- R² score ≥ 0.95
- MAE ≤ 10,000€  
- RMSE ≤ 15,000€

## Usage Examples

### Morning Training (Stop at 7 AM)
```bash
./scripts/run_loop_tuner_agent.sh catboost --end-time 07:00
```

### Quick Experimentation (50 trials)
```bash
./scripts/run_loop_tuner_agent.sh xgboost --max-trials 50
```

### Overnight Training (8 hours)
```bash
./scripts/run_loop_tuner_agent.sh lightgbm --duration-hours 8
```

### Continuous Optimization
```bash
./scripts/run_loop_tuner_agent.sh stack_ensemble --no-time-limit
```

## Notes
- Only one termination condition can be specified per run
- The script validates that exactly one termination condition is provided
- Training progress is logged and saved to CosmosDB
- All model types use Optuna for hyperparameter optimization
