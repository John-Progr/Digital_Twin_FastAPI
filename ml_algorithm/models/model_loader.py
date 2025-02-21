# model_loader.py
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor
from .exceptions import ModelNotFoundException, PredictionError
from .constants import ModelConstants
from typing import List
import pandas as pd
# Configure logger
logger = logging.getLogger(__name__)

# --- Model Loading Function ---
def load_rf_model(model_path: str) -> RandomForestRegressor:
    """
    Loads the Random Forest model from the specified .pkl file.

    :param model_path: Path to the pre-trained Random Forest model.
    :return: The loaded Random Forest model.
    :raises: ModelNotFoundException if the model can't be loaded.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            logger.info(f"Loaded model type: {type(model)}")

            # Check if the loaded model has the 'predict' method (validating it's a scikit-learn model)
            if not hasattr(model, 'predict'):
                raise ValueError("Loaded object is not a valid model with 'predict' method")

            return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise ModelNotFoundException(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ModelNotFoundException(f"Error loading model: {str(e)}")

def predict_overhead(model, features: List[float]) -> float:
    """
    Makes a prediction using the Random Forest model.

    :param model: The loaded Random Forest model.
    :param features: List of input features for the prediction.
    :return: The predicted value.
    :raises: PredictionError if the prediction fails.
    """
    try:
        # Create DataFrame with proper feature names
        feature_names = ['HELLO_INTERVAL', 'TC_INTERVAL', 'WINDOW_SIZE', 'AVG_NEIGHBORS', 'STD_NEIGHBORS']
        features_df = pd.DataFrame([features], columns=feature_names)


        # Ensure the feature order matches the model training
        if hasattr(model, "feature_names_in_"):
            features_df = features_df[model.feature_names_in_]

        # Debugging logs
        logger.info(f"Feature shape for prediction: {features_df.shape}")
        logger.info(f"Feature values: {features_df.to_dict(orient='records')}")
        prediction = model.predict(features_df.values)[0]
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise PredictionError(f"Error during prediction: {str(e)}")

# --- RCL Calculation Function ---
def calculate_rcl(h: float, t: float, w: float, avg_neighbors: float, std_neighbors: float) -> float:
    """
    Calculates RCL based on predefined coefficients.

    :param h: hello_interval
    :param t: tc_interval
    :param w: window_size
    :param avg_neighbors: avg_neighbors
    :param std_neighbors: std_neighbors
    :return: Calculated RCL value.
    """
    try:
        coef = ModelConstants.RCL_COEFFICIENTS
        rcl_value = (coef['intercept'] +
                     coef['hello'] * h +
                     coef['tc'] * t +
                     coef['window'] * w +
                     coef['avg_neighbors'] * avg_neighbors +
                     coef['std_neighbors'] * std_neighbors)
        return rcl_value
    except Exception as e:
        logger.error(f"Error during RCL calculation: {str(e)}")
        raise PredictionError(f"Error during RCL calculation: {str(e)}")
