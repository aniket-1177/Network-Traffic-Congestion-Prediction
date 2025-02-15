import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from ..utils.logger import get_logger

logger = get_logger()

def plot_class_distribution(df, output_dir='visualization'):
    """Plot the distribution of congestion classes"""
    logger.info("Plotting class distribution")
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='congestion', data=df)
    plt.title('Congestion Class Distribution')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved class distribution plot to {output_path}")

def plot_traffic_volume(df, output_dir='visualization'):
    """Plot traffic volume over time"""
    logger.info("Plotting traffic volume over time")
    
    # Extract hour and day
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.date
    
    # Aggregate traffic by hour and day
    traffic_volume = df.groupby(['day', 'hour'])['bytes_sent'].sum()
    
    plt.figure(figsize=(12, 8))
    traffic_volume.plot(kind='line', label='Traffic Volume')
    
    # Calculate and plot rolling average
    traffic_volume_rolling = traffic_volume.rolling(window=24).mean()
    traffic_volume_rolling.plot(kind='line', label='24-Hour Rolling Average')
    
    plt.title('Traffic Volume Over Time (Daily)')
    plt.xlabel('Hour')
    plt.ylabel('Bytes Sent')
    plt.legend()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'traffic_volume.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved traffic volume plot to {output_path}")

def plot_protocol_distribution(df, output_dir='visualization'):
    """Plot distribution of protocols"""
    logger.info("Plotting protocol distribution")
    
    protocol_counts = df['protocol'].value_counts()
    
    plt.figure(figsize=(8, 6))
    protocol_counts.plot(kind='bar')
    plt.title('Protocol Distribution')
    plt.xlabel('Protocol')
    plt.ylabel('Count')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'protocol_distribution.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved protocol distribution plot to {output_path}")

def plot_congestion_over_time(df, output_dir='visualization'):
    """Plot congestion rate over time"""
    logger.info("Plotting congestion over time")
    
    # Extract hour and day if not already present
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'day' not in df.columns:
        df['day'] = df['timestamp'].dt.date
    
    # Aggregate congestion by hour and day
    congestion_over_time = df.groupby(['day', 'hour'])['congestion'].mean()
    
    plt.figure(figsize=(12, 8))
    congestion_over_time.plot(kind='line', color='red')
    plt.title('Congestion over Time (Daily)')
    plt.xlabel('Hour')
    plt.ylabel('Congestion Rate')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'congestion_over_time.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved congestion over time plot to {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_dir='visualization'):
    """Plot confusion matrix for model evaluation"""
    logger.info("Plotting confusion matrix")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved confusion matrix plot to {output_path}")

def plot_feature_importance(model, feature_names, output_dir='visualization'):
    """Plot feature importance from the trained model"""
    logger.info("Plotting feature importance")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved feature importance plot to {output_path}")
    
    return importance_df

if __name__ == "__main__":
    # Test the visualization functions
    from ..data.generate_data import generate_traffic_data
    
    df = generate_traffic_data(1000)
    plot_class_distribution(df)
    plot_traffic_volume(df)
    plot_protocol_distribution(df)
    plot_congestion_over_time(df)