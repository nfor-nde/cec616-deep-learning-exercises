
Install dependencies

bash: pip install -r requirements.txt


Quick Setup for Your Iris Project
Here's the complete workflow for your Iris Flask app:

# 1. Create project folder
mkdir iris-flask-app
cd iris-flask-app

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Open in VS Code
code .

# 6. In VS Code, select the interpreter:
# - Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
# - Type "Python: Select Interpreter"
# - Choose the one with "./venv" in the path


Verify Your Environment
Create a test script check_env.py:

import sys
import sklearn
import flask
import numpy

print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"Flask version: {flask.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {numpy.__version__}")

Run it:

python check_env.py




Train the model

bash: python train_model.py


Run the Flask app

bash: python app.py

Access the app:

Open browser and go to http://localhost:5000

Use the web interface to make predictions

Or use the API endpoint with JSON:

python
        import requests
        import json

        url = 'http://localhost:5000/api/predict'
        data = {
            'features': [5.1, 3.5, 1.4, 0.2]  # Example: sepal_length, sepal_width, petal_length, petal_width
        }
        response = requests.post(url, json=data)
        print(response.json())
