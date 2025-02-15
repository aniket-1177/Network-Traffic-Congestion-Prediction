import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler
import pickle
import os
import yaml
from ..utils.logger import get_logger

logger = get_logger()

def ip_to_int(ip_address):
    """Convert IP address string to integer"""
    try:
        return int(ipaddress.IPv4Address(ip_address))
    except ipaddress.AddressValueError:
        return 0

def engineer_features(df, is_training=True, scaler_path='models/scaler.pkl', config_path='config/config.yaml'):
    """
    Perform feature engineering on the input dataframe
    
    Args:
        df: Input dataframe
        is_training: Whether this is for training (True) or inference (False)
        scaler_path: Path to save/load the scaler
        config_path: Path to config file
        
    Returns:
        Processed dataframe with engineered features
    """
    logger.info("Starting feature engineering process")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract timestamp features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.date
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Convert IPs to integers
    logger.info("Converting IP addresses to integers")
    df['source_ip_int'] = df['source_ip'].apply(ip_to_int)
    df['dest_ip_int'] = df['dest_ip'].apply(ip_to_int)
    
    # Aggregate features
    logger.info("Creating aggregate features")
    df['packet_size_mean'] = df.groupby(['source_ip_int', 'dest_ip_int'])['packet_size'].transform('mean')
    df['bytes_sent_sum'] = df.groupby(['source_ip_int', 'dest_ip_int'])['bytes_sent'].transform('sum')
    
    # Calculate packet rate (only if we have multiple records for time-based calculation)
    if len(df) > 1:
        logger.info("Calculating packet rate")
        df['packet_rate'] = df.groupby(['source_ip_int', 'dest_ip_int'])['packet_size'].transform(
            lambda x: x.rolling(window=5).mean().shift(1) / df['timestamp'].dt.second)
        df['packet_rate'] = df['packet_rate'].fillna(0)
    else:
        logger.info("Skipping packet rate calculation for single record")
        df['packet_rate'] = 0
    
    # Convert timestamp to seconds
    df['timestamp_seconds'] = df['timestamp'].astype(np.int64) // 10**9
    
    # One-hot encode categorical variables
    logger.info("One-hot encoding categorical features")
    df = pd.get_dummies(df, columns=['protocol'])
    
    # Drop unnecessary columns
    df = df.drop(['source_ip', 'dest_ip', 'day'], axis=1)
    
    # Handle timestamp column
    df = df.drop('timestamp', axis=1)
    
    # Get numerical features from config
    numerical_features = config['features']['numerical_features']
    
    # Scaling numerical features
    logger.info("Scaling numerical features")
    if is_training:
        # Create directory for scaler if it doesn't exist
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        
        # Save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
    else:
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        df[numerical_features] = scaler.transform(df[numerical_features])
        logger.info(f"Loaded scaler from {scaler_path}")
    
    logger.info("Feature engineering completed")
    return df

if __name__ == "__main__":
    # Test the function with sample data
    from ..data.generate_data import generate_traffic_data
    
    df = generate_traffic_data(100)
    processed_df = engineer_features(df)
    print(processed_df.head())