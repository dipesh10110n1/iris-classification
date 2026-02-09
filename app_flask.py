"""
app_flask.py
Flask API for Iris Flower Classification
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('iris_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
target_names = model_data['target_names']

# HTML template for testing
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Flower Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .form-group { margin: 15px 0; }
        label { display: inline-block; width: 150px; }
        input { padding: 5px; width: 200px; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        #result { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🌸 Iris Flower Classifier</h1>
    <form id="predictionForm">
        <div class="form-group">
            <label>Sepal Length (cm):</label>
            <input type="number" step="0.1" name="sepal_length" required>
        </div>
        <div class="form-group">
            <label>Sepal Width (cm):</label>
            <input type="number" step="0.1" name="sepal_width" required>
        </div>
        <div class="form-group">
            <label>Petal Length (cm):</label>
            <input type="number" step="0.1" name="petal_length" required>
        </div>
        <div class="form-group">
            <label>Petal Width (cm):</label>
            <input type="number" step="0.1" name="petal_width" required>
        </div>
        <button type="submit">Predict Species</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {
                sepal_length: parseFloat(formData.get('sepal_length')),
                sepal_width: parseFloat(formData.get('sepal_width')),
                petal_length: parseFloat(formData.get('petal_length')),
                petal_width: parseFloat(formData.get('petal_width'))
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            document.getElementById('result').innerHTML = `
                <h3>Prediction Result</h3>
                <p><strong>Species:</strong> ${result.species}</p>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                <p><strong>All Probabilities:</strong></p>
                <ul>
                    ${result.probabilities.map(p => `<li>${p.species}: ${(p.probability * 100).toFixed(2)}%</li>`).join('')}
                </ul>
            `;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the home page with prediction form"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict iris species from flower measurements
    
    Expected JSON format:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    try:
        data = request.get_json()
        
        # Extract features
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        response = {
            'species': target_names[prediction],
            'species_id': int(prediction),
            'confidence': float(probabilities[prediction]),
            'probabilities': [
                {'species': target_names[i], 'probability': float(prob)}
                for i, prob in enumerate(probabilities)
            ],
            'input_features': {
                feature_names[i]: float(data[key])
                for i, key in enumerate(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            }
        }
        
        return jsonify(response), 200
        
    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the model"""
    return jsonify({
        'model_type': 'Logistic Regression',
        'features': feature_names,
        'classes': target_names,
        'accuracy': float(model_data['accuracy'])
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True}), 200

if __name__ == '__main__':
    print("🌸 Starting Flask Iris Classifier API...")
    print("📍 API available at: http://localhost:5002")
    print("📝 Documentation:")
    print("   - Web Interface: http://localhost:5002/")
    print("   - Predict: POST http://localhost:5002/predict")
    print("   - Model Info: GET http://localhost:5002/model-info")
    print("   - Health: GET http://localhost:5002/health")
    app.run(debug=True, host='0.0.0.0', port=5002)
