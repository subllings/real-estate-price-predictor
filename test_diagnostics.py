#!/usr/bin/env python3
"""
Test des diagnostics de généralisation
"""

def analyze_generalization(r2_train, r2_test):
    gap = r2_train - r2_test
    # Logique alignée avec train_test_metrics_logger.py
    if gap < 0:
        return "Possible underfitting"
    elif gap < 0.05:
        return "Excellent generalization"
    elif gap < 0.08:
        return "Good generalization"
    elif gap < 0.12:
        return "Moderate overfitting"
    else:
        return "Strong overfitting"

# Test avec les valeurs de l'interface visible
test_cases = [
    (0.845600, 0.823400, "R² Train=0.845600, R² Test=0.823400"),  # Rang 1
    (0.939787, 0.793661, "R² Train=0.939787, R² Test=0.793661"),  # Rang 2
    (0.834276, 0.790394, "R² Train=0.834276, R² Test=0.790394"),  # Rang 3
    (0.837934, 0.782997, "R² Train=0.837934, R² Test=0.782997"),  # Rang 4
]

print("Test des diagnostics de généralisation:")
print("=" * 50)

for r2_train, r2_test, description in test_cases:
    gap = r2_train - r2_test
    diagnostic = analyze_generalization(r2_train, r2_test)
    print(f"{description}")
    print(f"  R² Gap: {gap:.6f}")
    print(f"  Diagnostic: {diagnostic}")
    print()
