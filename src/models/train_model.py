import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
import pickle
import os
import yaml
from ..utils.logger import get_logger

logger = get_logger()

def train_model(X, y, config_path='config/config.yaml'):
    """
    Train the traffic congestion prediction model
    
    Args:
        X: Feature dataframe
        y: Target variable
        config_path: Path to configuration file
        
    Returns:
        Trained model, test data, and test predictions
    """
    logger.info("Starting model training process")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model paths
    model_path = config['paths']['model_path']
    features_path = config['paths']['features_path']
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    
    # Split data
    logger.info("Splitting data into train and test sets")
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Save feature dataframe for later use in prediction
    X.to_pickle(features_path)
    logger.info(f"Saved feature dataframe to {features_path}")
    
    # Time Series Cross-Validation
    logger.info("Performing time series cross-validation")
    cv_splits = config['model']['cv_splits']
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Get hyperparameter grid
    param_grid = config['model']['param_grid']
    n_iter = config['model']['search_iterations']
    
    best_f1 = 0
    best_model = None
    
    for train_index, test_index in tscv.split(X):
        logger.info("Training on fold")
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        model = RandomForestClassifier(random_state=random_state)
        grid_search = RandomizedSearchCV(
            model, param_grid, cv=3, scoring='f1', n_iter=n_iter
        )
        
        grid_search.fit(X_train_fold, y_train_fold)
        current_model = grid_search.best_estimator_
        
        y_pred_fold = current_model.predict(X_test_fold)
        f1 = f1_score(y_test_fold, y_pred_fold)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = current_model
            logger.info(f"New best model found with F1 score: {best_f1:.4f}")
    
    # Final evaluation
    logger.info("Evaluating final model on test set")
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    
    logger.info(f"Final model metrics:")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Saved model to {model_path}")
    
    return best_model, X_test, y_test, y_pred

if __name__ == "__main__":
    # Test the function with sample data
    from ..data.generate_data import generate_traffic_data
    from ..features.feature_engineering import engineer_features
    
    df = generate_traffic_data(1000)
    processed_df = engineer_features(df)
    
    X = processed_df.drop('congestion', axis=1)
    y = processed_df['congestion']
    
    model, X_test, y_test, y_pred = train_model(X, y)
    print(classification_report(y_test, y_pred))