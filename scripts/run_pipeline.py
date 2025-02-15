import os
import sys
import yaml
import argparse
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generate_data import generate_traffic_data
from src.features.feature_engineering import engineer_features
from src.models.train_model import train_model
from src.visualization.visualize import (
    plot_class_distribution, plot_traffic_volume, 
    plot_protocol_distribution, plot_congestion_over_time,
    plot_confusion_matrix, plot_feature_importance
)
from src.utils.logger import get_logger

logger = get_logger()

def run_pipeline(config_path='config/config.yaml', generate_new_data=False):
    """
    Run the complete traffic analysis pipeline
    
    Args:
        config_path: Path to configuration file
        generate_new_data: Whether to generate new data or use existing
    """
    logger.info("Starting traffic analysis pipeline")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Data Generation/Loading
    raw_data_path = config['data']['raw_path']
    
    if generate_new_data or not os.path.exists(raw_data_path):
        logger.info("Generating new traffic data")
        df = generate_traffic_data(n_samples=1000, output_path=raw_data_path)
    else:
        logger.info(f"Loading existing data from {raw_data_path}")
        df = pd.read_csv(raw_data_path, parse_dates=['timestamp'])
    
    # Step 2: Exploratory Data Analysis
    logger.info("Performing exploratory data analysis")
    plot_class_distribution(df)
    plot_traffic_volume(df)
    plot_protocol_distribution(df)
    plot_congestion_over_time(df)
    
    # Step 3: Feature Engineering
    logger.info("Performing feature engineering")
    processed_df = engineer_features(df)
    
    # Step 4: Model Training
    logger.info("Training the model")
    X = processed_df.drop('congestion', axis=1)
    y = processed_df['congestion']
    
    model, X_test, y_test, y_pred = train_model(X, y, config_path=config_path)
    
    # Step 5: Model Evaluation Visualization
    logger.info("Creating model evaluation visualizations")
    plot_confusion_matrix(y_test, y_pred)
    importance_df = plot_feature_importance(model, X.columns)
    
    logger.info("Pipeline completed successfully")
    return model, importance_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the traffic analysis pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate new data instead of using existing')

    args = parser.parse_args()
    
    # Run the pipeline
    model, importance_df = run_pipeline(
        config_path=args.config,
        generate_new_data=args.generate_data
    )
    
    # Print top 5 important features
    print("\nTop 5 Important Features:")
    print(importance_df.head(5))