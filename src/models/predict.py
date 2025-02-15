import pandas as pd
import numpy as np
import pickle
import yaml
import os
from ..features.feature_engineering import ip_to_int, engineer_features
from ..utils.logger import get_logger

logger = get_logger()

def load_artifacts(config_path='config/config.yaml'):
    """Load all necessary artifacts for prediction"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = config['paths']['model_path']
    scaler_path = config['paths']['scaler_path']
    features_path = config['paths']['features_path']
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        X = pd.read_pickle(features_path)
        
        return model, scaler, X
    except FileNotFoundError as e:
        logger.error(f"Error loading artifacts: {e}")
        raise

def predict_congestion(input_data, config_path='config/config.yaml'):
    """
    Predicts traffic congestion using the saved model.
    
    Args:
        input_data (dict): A dictionary containing the input features for prediction.
        config_path (str): Path to the configuration file.
                          
    Returns:
        int: The predicted congestion level (0 or 1).
        or
        str: An error message if input is invalid.
    """
    try:
        logger.info("Starting prediction process")
        
        # Load model and artifacts
        model, scaler, X = load_artifacts(config_path)
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Convert timestamp_seconds to timestamp if needed
        if 'timestamp_seconds' in input_df.columns and 'timestamp' not in input_df.columns:
            input_df['timestamp'] = pd.to_datetime(input_df['timestamp_seconds'], unit='s')
        
        # Perform feature engineering (will handle scaling)
        processed_input = engineer_features(
            input_df, 
            is_training=False,
            scaler_path=os.path.join(os.path.dirname(config_path), '..', 
                                    'models/scaler.pkl')
        )
        
        # Handle potential missing columns after preprocessing
        for col in X.columns:
            if col not in processed_input.columns:
                processed_input[col] = 0
        
        # Ensure correct column order
        processed_input = processed_input[X.columns]
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        logger.info(f"Prediction completed: {prediction}")
        
        return int(prediction)
    
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        return f"Invalid input data: {e}"
    except KeyError as e:
        logger.error(f"Missing key in input data: {e}")
        return f"Missing key in input data: {e}"
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        return "Model or scaler file not found."
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    # Test the prediction function
    input_data = {
        'packet_size': 1200,
        'bytes_sent': 4800,
        'hour': 17,
        'source_ip': '192.168.1.10',
        'dest_ip': '10.0.0.20',
        'protocol': 'TCP',
        'timestamp_seconds': 1700000000,
    }
    
    result = predict_congestion(input_data)
    print(f"Predicted Congestion: {result}")