import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

print("[TEST] Starting isolated CatBoost test...")

# Génère des données factices
X = pd.DataFrame(np.random.rand(100, 10), columns=[f"f{i}" for i in range(10)])
y = pd.Series(np.random.rand(100))

# Initialiser un modèle simple (CPU only, 1 thread)
model = CatBoostRegressor(
    learning_rate=0.1,
    depth=6,
    task_type="CPU",
    thread_count=1,
    verbose=0
)

# Essayer de fitter le modèle
try:
    model.fit(X, y, eval_set=(X, y))
    print("✅ SUCCESS: CatBoost fit() worked without error.")
except Exception as e:
    print(f"❌ ERROR: CatBoost fit() failed: {e}")
