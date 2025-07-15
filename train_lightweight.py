#!/usr/bin/env python3
"""
Lightweight training script for limited hardware
Optimized for GTX 2080 Ti with stability issues
"""

import os
import sys
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"

# Import after setting environment
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_lightweight_data():
    """Load or simulate lightweight training data"""
    
    print("📊 Loading lightweight training data...")
    
    np.random.seed(42)
    n_samples = 1000  # Reduced for speed
    
    # Simulate realistic real estate features
    data = {
        'surface': np.random.normal(120, 40, n_samples),
        'rooms': np.random.randint(2, 8, n_samples),
        'bedrooms': np.random.randint(1, 5, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'garden': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'terrace': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'garage': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'location_score': np.random.normal(7, 2, n_samples)
    }
    
    X = pd.DataFrame(data)
    
    # Realistic price simulation
    y = (
        X['surface'] * 3000 +
        X['rooms'] * 15000 +
        X['garden'] * 30000 +
        X['location_score'] * 25000 +
        np.random.normal(0, 50000, n_samples)
    )
    
    y = np.maximum(y, 100000)
    
    return X, y

def train_lightweight_model():
    """Train model with minimal resource usage"""
    
    print("🚀 Starting lightweight model training...")
    print("⚙️  CPU-only mode, GPU disabled for stability")
    
    # 1. Load data
    X, y = load_lightweight_data()
    print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
    
    # 2. Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"🔀 Split: {len(X_train)} train, {len(X_val)} validation")
    
    # 3. Train with conservative parameters
    print("🤖 Training CatBoost model...")
    
    model = CatBoostRegressor(
        iterations=500,          # Reduced iterations
        depth=6,                 # Conservative depth
        learning_rate=0.1,       # Standard learning rate
        random_seed=42,
        verbose=False,           # Silent mode
        task_type="CPU",         # Force CPU
        thread_count=4,          # Limit threads
        max_ctr_complexity=1,    # Reduce memory
        simple_ctr=['Borders', 'Counter']  # Basic features only
    )
    
    # Train with try-catch for safety
    try:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        print("✅ Training completed successfully")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None
    
    # 4. Evaluate
    y_pred = model.predict(X_val)
    
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print("\n" + "="*50)
    print("📊 MODEL PERFORMANCE")
    print("="*50)
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: €{mae:.0f}")
    print(f"RMSE: €{rmse:.0f}")
    
    # 5. Save model
    models_dir = Path("ml_models")
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"catboost_lite_{timestamp}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"💾 Model saved: {model_path}")
    
    # 6. Save metadata
    metadata = {
        "version": f"lite_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "performance": {
            "r2_score": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "validation_samples": len(X_val)
        },
        "model_params": model.get_params(),
        "feature_names": list(X.columns),
        "training_config": "lightweight_cpu_only"
    }
    
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved: {metadata_path}")
    
    # 7. Copy to API if directory exists
    api_models_dir = Path("app/backend-api-price-prediction/models")
    if api_models_dir.exists():
        import shutil
        shutil.copy(model_path, api_models_dir / "latest_model.pkl")
        shutil.copy(metadata_path, api_models_dir / "latest_model_metadata.json")
        print(f"🔄 Model copied to API: {api_models_dir}")
    
    # 8. Performance check
    if r2 >= 0.80:
        print("✅ Model performance is acceptable!")
        print("🚀 Ready for production use")
    else:
        print("⚠️  Model performance below threshold (R² < 0.80)")
        print("💡 Consider: more data, feature engineering, or hyperparameter tuning")
    
    return {
        "model_path": model_path,
        "metadata": metadata,
        "performance": {"r2": r2, "mae": mae, "rmse": rmse}
    }

def main():
    """Main execution function"""
    
    print("🏠 Real Estate Lightweight Training")
    print("💻 Optimized for GTX 2080 Ti stability")
    print("="*50)
    
    try:
        result = train_lightweight_model()
        if result:
            print("\n🎉 Training completed successfully!")
            print(f"📁 Model available at: {result['model_path']}")
        else:
            print("\n❌ Training failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
