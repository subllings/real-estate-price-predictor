"""
Azure ML Training Script
Adapted version of the training script for Azure ML cloud execution
"""

import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Azure ML imports
from azureml.core import Run
import mlflow
import mlflow.catboost

# ML imports
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Azure ML Real Estate Training')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and artifacts')
    
    return parser.parse_args()


def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_data(config):
    """Load and prepare training data"""
    print("📊 Loading training data...")
    
    # Load data
    train_data = pd.read_parquet(config['data']['train_path'])
    test_data = pd.read_parquet(config['data']['test_path'])
    
    # Prepare features and target
    target_col = 'price'
    feature_cols = [col for col in train_data.columns if col != target_col]
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    # Create validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"✅ Data loaded:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features: {len(feature_cols)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def objective(trial, X_train, y_train, X_val, y_val, config, run_context):
    """Optuna objective function with Azure ML logging"""
    
    # Suggest hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', *config['hyperparameters']['learning_rate']),
        'depth': trial.suggest_int('depth', *config['hyperparameters']['depth']),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *config['hyperparameters']['l2_leaf_reg']),
        'iterations': trial.suggest_int('iterations', *config['hyperparameters']['iterations']),
        'random_seed': 42,
        'verbose': False
    }
    
    # Train model
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    # Log trial metrics to Azure ML
    run_context.log(f'trial_{trial.number}_r2_val', r2_val)
    run_context.log(f'trial_{trial.number}_mae_val', mae_val)
    run_context.log(f'trial_{trial.number}_rmse_val', rmse_val)
    
    # Log hyperparameters
    for param_name, param_value in params.items():
        run_context.log(f'trial_{trial.number}_{param_name}', param_value)
    
    return r2_val


def train_best_model(study, X_train, y_train, X_val, y_val, X_test, y_test, config, run_context):
    """Train final model with best hyperparameters"""
    
    print("🏆 Training final model with best hyperparameters...")
    
    # Get best parameters
    best_params = study.best_params
    best_params['random_seed'] = 42
    best_params['verbose'] = False
    
    print(f"📋 Best parameters: {best_params}")
    
    # Train final model
    best_model = CatBoostRegressor(**best_params)
    best_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=True)
    
    # Evaluate on all sets
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'r2_train': r2_score(y_train, y_train_pred),
        'r2_val': r2_score(y_val, y_val_pred),
        'r2_test': r2_score(y_test, y_test_pred),
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'mae_val': mean_absolute_error(y_val, y_val_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'rmse_val': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred))
    }
    
    # Calculate validation gap
    validation_gap = abs(metrics['r2_test'] - metrics['r2_val'])
    metrics['validation_gap'] = validation_gap
    
    # Log all metrics to Azure ML
    for metric_name, metric_value in metrics.items():
        run_context.log(metric_name, metric_value)
    
    # Quality gate evaluation
    quality_gates = config['quality_gates']
    quality_passed = (
        metrics['r2_test'] >= quality_gates['min_r2_test'] and
        validation_gap <= quality_gates['max_validation_gap']
    )
    
    run_context.log('quality_gate_passed', quality_passed)
    
    print(f"📊 Final Model Performance:")
    print(f"   R² Test: {metrics['r2_test']:.4f}")
    print(f"   R² Validation: {metrics['r2_val']:.4f}")
    print(f"   Validation Gap: {validation_gap:.4f}")
    print(f"   MAE Test: €{metrics['mae_test']:.0f}")
    print(f"   Quality Gate: {'✅ PASSED' if quality_passed else '❌ FAILED'}")
    
    if quality_passed:
        print("🎉 Model meets quality requirements!")
    else:
        print("⚠️  Model does not meet quality requirements")
        print(f"   Required R² ≥ {quality_gates['min_r2_test']}")
        print(f"   Required Gap ≤ {quality_gates['max_validation_gap']}")
    
    return best_model, metrics, quality_passed


def save_model_artifacts(model, metrics, feature_cols, output_dir, quality_passed):
    """Save model and metadata"""
    
    print("💾 Saving model artifacts...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / 'model.pkl'
    joblib.dump(model, model_path)
    
    # Save feature names
    features_path = output_path / 'features.json'
    with open(features_path, 'w') as f:
        json.dump({'features': feature_cols}, f, indent=2)
    
    # Save metrics
    metadata = {
        'metrics': metrics,
        'quality_gate_passed': quality_passed,
        'feature_count': len(feature_cols),
        'model_type': 'CatBoostRegressor'
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Artifacts saved to {output_dir}")
    print(f"   Model: {model_path}")
    print(f"   Features: {features_path}")
    print(f"   Metadata: {metadata_path}")
    
    return model_path, features_path, metadata_path


def main():
    """Main training function"""
    
    print("🚀 Starting Azure ML Real Estate Training")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Get Azure ML run context
    run = Run.get_context()
    
    # Enable MLflow tracking
    mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data(config)
    
    # Log dataset info
    run.log('n_features', len(feature_cols))
    run.log('n_train_samples', len(X_train))
    run.log('n_val_samples', len(X_val))
    run.log('n_test_samples', len(X_test))
    
    # Create Optuna study
    print("🔍 Starting hyperparameter optimization...")
    
    study = optuna.create_study(
        direction=config['optuna']['direction'],
        study_name=config['optuna']['study_name']
    )
    
    # Optimize
    n_trials = config['optuna']['n_trials']
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, config, run),
        n_trials=n_trials
    )
    
    # Log optimization results
    run.log('n_trials', n_trials)
    run.log('best_trial_value', study.best_value)
    
    print(f"✅ Optimization complete!")
    print(f"   Best R² (validation): {study.best_value:.4f}")
    print(f"   Trials completed: {len(study.trials)}")
    
    # Train final model
    with mlflow.start_run(nested=True):
        best_model, metrics, quality_passed = train_best_model(
            study, X_train, y_train, X_val, y_val, X_test, y_test, config, run
        )
        
        # Log model to MLflow
        mlflow.catboost.log_model(best_model, "model")
    
    # Save artifacts
    model_path, features_path, metadata_path = save_model_artifacts(
        best_model, metrics, feature_cols, args.output_dir, quality_passed
    )
    
    # Upload files to Azure ML
    run.upload_file('model.pkl', str(model_path))
    run.upload_file('features.json', str(features_path))
    run.upload_file('metadata.json', str(metadata_path))
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final R² Score: {metrics['r2_test']:.4f}")
    print(f"🎯 Quality Gate: {'✅ PASSED' if quality_passed else '❌ FAILED'}")


if __name__ == "__main__":
    main()
