from flask import Flask, request, render_template, jsonify
import os
import sys
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.predict import predict_congestion
from src.utils.logger import get_logger

application = Flask(__name__)
app = application

logger = get_logger()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        try:
            # Collect input data from the form
            input_data = {
                'packet_size': int(request.form.get('packet_size')),
                'bytes_sent': int(request.form.get('bytes_sent')),
                'source_ip': request.form.get('source_ip'),
                'dest_ip': request.form.get('dest_ip'),
                'protocol': request.form.get('protocol'),
                'timestamp_seconds': int(request.form.get('timestamp_seconds')),
                "hour": int(request.form.get('hour'))

            }
            
            # Make prediction
            result = predict_congestion(input_data)
            
            # Handle error message from prediction function
            if isinstance(result, str):
                return render_template('predict.html', error=result)
            
            # Display result on the same page
            congestion_status = "Congested" if result == 1 else "Normal"
            return render_template('predict.html', result=congestion_status)
        
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return render_template('predict.html', error="Invalid input or server error.")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
