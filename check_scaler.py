import joblib
import os
from sklearn.preprocessing import StandardScaler

scaler_path = "models/target_scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"Scaler type: {type(scaler)}")
    print(f"Is fitted (has mean_): {hasattr(scaler, 'mean_')}")
    if hasattr(scaler, 'mean_'):
        print(f"Mean: {scaler.mean_}")
        print(f"Scale: {scaler.scale_}")
else:
    print("Scaler file not found.")
