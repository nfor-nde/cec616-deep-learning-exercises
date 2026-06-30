from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model and class names
model_path = 'iris_model.pkl'
classes_path = 'class_names.pkl'

if os.path.exists(model_path) and os.path.exists(classes_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(classes_path, 'rb') as f:
        class_names = pickle.load(f)
    print("Model loaded successfully!")
else:
    print("Model files not found. Please run train_model.py first.")
    model = None
    class_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
            features = data.get('features', [])
        else:
            # Get form data
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        
        # Get class name
        predicted_class = class_names[prediction]
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'class_name': predicted_class,
            'probabilities': {
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            }
        }
        
        if request.is_json:
            return jsonify(response)
        else:
            return render_template('result.html', result=response)
            
    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e)
        }
        if request.is_json:
            return jsonify(error_response), 400
        else:
            return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    return predict()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)