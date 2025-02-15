import pandas as pd
import numpy as np
import os
from ..utils.logger import get_logger

logger = get_logger()

def traffic_pattern(hour):
    """Generate realistic traffic patterns based on hour of day"""
    if 8 <= hour <= 12 or 16 <= hour <= 20:  # Peak hours
        return np.random.randint(500, 1500)
    else:  # Off-peak
        return np.random.randint(100, 500)

def generate_traffic_data(n_samples=1000, output_path='data/traffic_data.csv'):
    """
    Generate synthetic network traffic data and save to CSV
    
    Args:
        n_samples: Number of data points to generate
        output_path: Path to save the CSV file
    """
    logger.info(f"Generating {n_samples} samples of synthetic traffic data")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate timestamps
    start_date = pd.to_datetime('2025-01-01')
    timestamps = start_date + pd.to_timedelta(np.random.randint(0, 86400 * 7, n_samples), unit='s')
    
    # Generate traffic features
    packet_sizes = [traffic_pattern(ts.hour) for ts in timestamps]
    bytes_sent = [size * np.random.randint(1, 5) for size in packet_sizes]
    source_ips = [f"192.168.1.{i}" for i in np.random.randint(1, 255, n_samples)]
    dest_ips = [f"10.0.0.{i}" for i in np.random.randint(1, 255, n_samples)]
    protocols = np.random.choice(["TCP", "UDP", "HTTP"], n_samples)
    congestion = np.random.randint(0, 2, n_samples)
    
    # Create DataFrame
    df_generated = pd.DataFrame({
        "timestamp": timestamps,
        "source_ip": source_ips,
        "dest_ip": dest_ips,
        "protocol": protocols,
        "packet_size": packet_sizes,
        "bytes_sent": bytes_sent,
        "congestion": congestion
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df_generated.to_csv(output_path, index=False)
    logger.info(f"Saved generated data to {output_path}")
    
    return df_generated

if __name__ == "__main__":
    generate_traffic_data()