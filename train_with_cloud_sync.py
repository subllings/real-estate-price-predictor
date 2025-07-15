#!/usr/bin/env python3
"""
Integration script to bridge local training with cloud architecture
This script can be run from your laptop to train models and sync with cloud
"""

import os
import sys
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from agents.cloud_training_agent import CloudTrainingAgent

# Check if we have access to your existing data processing
try:
    # Try to import your existing data processing if available
    from utils.data_preprocessing import load_and_preprocess_data
    from utils.feature_engineering import create_features
    DATA_UTILS_AVAILABLE = True
except ImportError:
    print("ℹ️  Custom data utils not found, using basic data simulation")
    DATA_UTILS_AVAILABLE = False


def load_sample_data():
    """Load or simulate training data"""
    
    if DATA_UTILS_AVAILABLE:
        try:
            # Use your existing data loading
            X, y, feature_names = load_and_preprocess_data()
            return X, y, feature_names
        except Exception as e:
            print(f"⚠️  Error loading real data: {e}")
    
    # Fallback to simulated data
    print("📊 Using simulated real estate data")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Simulate realistic real estate features
    data = {
        'surface': np.random.normal(120, 40, n_samples),  # m²
        'rooms': np.random.randint(2, 8, n_samples),
        'bedrooms': np.random.randint(1, 5, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'garden': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'terrace': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'pool': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'garage': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'fireplace': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'location_score': np.random.normal(7, 2, n_samples)  # 1-10 score
    }
    
    X = pd.DataFrame(data)
    
    # Simulate realistic price based on features
    y = (
        X['surface'] * 3000 +  # €3000 per m²
        X['rooms'] * 15000 +   # €15k per room
        X['garden'] * 30000 +  # Garden adds €30k
        X['pool'] * 50000 +    # Pool adds €50k
        X['location_score'] * 25000 +  # Location multiplier
        np.random.normal(0, 50000, n_samples)  # Random noise
    )
    
    # Ensure positive prices
    y = np.maximum(y, 100000)
    
    feature_names = list(X.columns)
    
    return X.values, y, feature_names


async def train_and_sync_model():
    """Main function to train model and sync with cloud"""
    
    print("🚀 Starting enhanced model training with cloud sync...")
    
    # 1. Load training data
    X, y, feature_names = load_sample_data()
    print(f"📊 Loaded {len(X)} samples with {len(feature_names)} features")
    
    # 2. Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"🔀 Split: {len(X_train)} train, {len(X_val)} validation")
    
    # 3. Initialize training agent
    agent = CloudTrainingAgent()
    
    # 4. Train with cloud integration
    result = await agent.train_with_cloud_logging(
        X_train, y_train, X_val, y_val, feature_names
    )
    
    # 5. Display results
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED")
    print("="*60)
    print(f"Model Version: {result['version']}")
    print(f"R² Score: {result['metadata']['performance']['r2_score']:.4f}")
    print(f"MAE: €{result['metadata']['performance']['mae']:.0f}")
    print(f"RMSE: €{result['metadata']['performance']['rmse']:.0f}")
    print(f"Local Path: {result['local_path']}")
    
    # 6. Check if model is production-ready
    if result['metadata']['deployment_ready']:
        print("✅ Model meets production threshold!")
        print("🚀 Ready for deployment to API")
        
        # Copy to your existing API folder if it exists
        api_models_dir = Path("app/backend-api-price-prediction/models")
        if api_models_dir.exists():
            import shutil
            shutil.copy(result['local_path'], api_models_dir / "latest_model.pkl")
            
            # Also copy metadata
            metadata_path = result['local_path'].with_suffix('_metadata.json')
            if metadata_path.exists():
                shutil.copy(metadata_path, api_models_dir / "latest_model_metadata.json")
                
            print(f"📁 Model copied to API directory: {api_models_dir}")
        else:
            print(f"ℹ️  API directory not found. Model available at: {result['local_path']}")
    else:
        print("⚠️  Model below production threshold. Needs improvement.")
    
    # 7. Generate summary report
    generate_training_report(result)
    
    return result


def generate_training_report(result):
    """Generate a training report"""
    
    report = f"""
# Training Report - {result['version']}

## Model Performance
- **R² Score**: {result['metadata']['performance']['r2_score']:.4f}
- **Mean Absolute Error**: €{result['metadata']['performance']['mae']:.0f}
- **Root Mean Square Error**: €{result['metadata']['performance']['rmse']:.0f}
- **Validation Samples**: {result['metadata']['performance']['validation_samples']}

## Hyperparameters (Best)
"""
    
    for key, value in result['metadata']['optuna_study']['best_params'].items():
        report += f"- **{key}**: {value}\n"
    
    report += f"""
## Optimization Details
- **Total Trials**: {result['metadata']['optuna_study']['n_trials']}
- **Best Trial**: #{result['metadata']['optuna_study']['best_trial']}
- **Study Name**: {result['metadata']['optuna_study']['study_name']}

## Feature Importance (Top 5)
"""
    
    # Sort features by importance
    feature_importance = result['metadata']['feature_importance']
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    for feature, importance in sorted_features[:5]:
        report += f"- **{feature}**: {importance:.4f}\n"
    
    report += f"""
## Deployment Status
- **Production Ready**: {'✅ Yes' if result['metadata']['deployment_ready'] else '❌ No'}
- **Status**: {result['metadata']['status']}
- **Created**: {result['metadata']['created_at']}

## Next Steps
{'- Deploy to production API' if result['metadata']['deployment_ready'] else '- Improve model performance'}
- Monitor performance in production
- Schedule retraining based on new data
"""
    
    # Save report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / f"training_report_{result['version']}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Training report saved: {report_path}")


if __name__ == "__main__":
    print("🏠 Real Estate Model Training with Cloud Integration")
    print("=" * 60)
    
    # Check environment
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Copy .env.template to .env and configure Azure credentials")
        print("ℹ️  Training will proceed with local-only mode")
    
    # Run training
    try:
        result = asyncio.run(train_and_sync_model())
        print("\n🎉 All done! Check the reports folder for detailed results.")
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
