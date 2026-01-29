import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
import streamlit as st
import os
import warnings


@st.cache_resource
def load_models():
    """
    Load all saved models and dependencies
    Returns: dict with all model components
    """
    models_dir = 'models'

    # Suppress sklearn version warnings (use at your own risk)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    try:
        # Load Random Forest
        print("Loading Random Forest model...")
        with open(f'{models_dir}/rf_enhanced_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print("✓ RF model loaded")

        # Load Scaler
        print("Loading Scaler...")
        with open(f'{models_dir}/scaler_enhanced.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded")

        # Load Feature Names
        print("Loading Feature Names...")
        with open(f'{models_dir}/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✓ Feature names loaded ({len(feature_names)} features)")

        # Load Risk Mappings
        print("Loading Risk Mappings...")
        with open(f'{models_dir}/risk_mappings.pkl', 'rb') as f:
            risk_mappings = pickle.load(f)
        print("✓ Risk mappings loaded")

        # Load ANN (optional)
        ann_model = None
        ann_path = f'{models_dir}/ann_enhanced_model.h5'
        if os.path.exists(ann_path):
            print("Loading ANN model...")
            try:
                ann_model = keras.models.load_model(ann_path)
                print("✓ ANN model loaded")
            except Exception as e:
                print(f"⚠ Could not load ANN model: {e}")
                ann_model = None

        return {
            'rf_model': rf_model,
            'ann_model': ann_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'risk_mappings': risk_mappings
        }

    except FileNotFoundError as e:
        print(f"⚠️ Model file not found: {e}")
        st.error(f"⚠️ Model file not found: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        import traceback
        print(traceback.format_exc())
        st.error(f"⚠️ Error loading models: {e}")
        return None


@st.cache_data
def load_data():
    """
    Load the labeled dataset
    """
    try:
        df = pd.read_csv('data/canteen_labeled_enhanced.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please ensure 'canteen_labeled_enhanced.csv' is in the 'data/' folder.")
        return None


def predict_risk(input_data, models_dict, use_ann=False):
    """
    Make risk prediction using trained model

    Parameters:
    -----------
    input_data : dict
        Dictionary of feature values
    models_dict : dict
        Dictionary containing loaded models
    use_ann : bool
        Whether to use ANN instead of RF

    Returns:
    --------
    prediction : dict
        Contains risk_level, risk_label, risk_score, probabilities
    """
    # Extract models
    model = models_dict['ann_model'] if use_ann else models_dict['rf_model']
    scaler = models_dict['scaler']
    feature_names = models_dict['feature_names']
    risk_mappings = models_dict['risk_mappings']

    # Create dataframe from input
    df_input = pd.DataFrame([input_data])

    # Ensure all features are present
    for feature in feature_names:
        if feature not in df_input.columns:
            df_input[feature] = 0  # Default value

    # Select and order features
    df_input = df_input[feature_names]

    # Scale features
    X_scaled = scaler.transform(df_input)

    # Make prediction
    if use_ann:
        # ANN prediction
        y_pred_prob = model.predict(X_scaled, verbose=0)
        y_pred_class = np.argmax(y_pred_prob, axis=1)[0]
        risk_level = y_pred_class + 1  # Convert 0-3 to 1-4
        probabilities = y_pred_prob[0].tolist()
    else:
        # Random Forest prediction
        y_pred = model.predict(df_input)
        y_pred_prob = model.predict_proba(df_input)

        # Convert prediction to risk level
        risk_label = y_pred[0]
        risk_level = risk_mappings['risk_mapping_reverse'][risk_label]
        probabilities = y_pred_prob[0].tolist()

    # Get risk label
    risk_label = risk_mappings['risk_level_to_label'][risk_level]

    # Calculate risk score (0-100)
    risk_score = input_data.get('TOTAL_RISK_SCORE', 0)

    return {
        'risk_level': risk_level,
        'risk_label': risk_label,
        'risk_score': risk_score,
        'probabilities': probabilities,
        'confidence': max(probabilities) * 100
    }


def get_risk_color(risk_label):
    """Get color code for risk level"""
    colors = {
        'VERY_LOW': '#2ecc71',
        'LOW': '#3498db',
        'MEDIUM': '#f39c12',
        'HIGH': '#e74c3c',
        'CRITICAL': '#c0392b'
    }
    return colors.get(risk_label, '#95a5a6')


def get_risk_icon(risk_label):
    """Get emoji icon for risk level"""
    icons = {
        'VERY_LOW': '✅',
        'LOW': '🟢',
        'MEDIUM': '🟡',
        'HIGH': '🔴',
        'CRITICAL': '🚨'
    }
    return icons.get(risk_label, '⚪')