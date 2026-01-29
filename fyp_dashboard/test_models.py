from utils.helpers import load_models, load_data
import pandas as pd

print("Testing model loading...")

# Load models
models = load_models()
if models:
    print("✓ Models loaded successfully!")
    print(f"  - RF Model: {type(models['rf_model'])}")
    print(f"  - Scaler: {type(models['scaler'])}")
    print(f"  - Features: {len(models['feature_names'])} features")
else:
    print("✗ Failed to load models")

# Load data
df = load_data()
if df is not None:
    print(f"\n✓ Data loaded successfully!")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
else:
    print("\n✗ Failed to load data")

# Test prediction
if models and df is not None:
    print("\n Testing prediction on first row...")

    # Get first row's features
    first_row = df.iloc[0]
    input_data = first_row[models['feature_names']].to_dict()

    from utils.helpers import predict_risk

    result = predict_risk(input_data, models, use_ann=False)
    print(f"  - Predicted Risk: {result['risk_label']}")
    print(f"  - Risk Score: {result['risk_score']:.2f}")
    print(f"  - Confidence: {result['confidence']:.2f}%")
    print("\n✓ All tests passed!")