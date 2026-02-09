"""
app_fastapi.py
FastAPI API for Iris Flower Classification
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import List
import uvicorn

# Load the model
with open('iris_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
target_names = model_data['target_names']

# Initialize FastAPI app
app = FastAPI(
    title="Iris Flower Classifier API",
    description="Classify iris flowers using machine learning",
    version="1.0.0"
)

# Pydantic models for request/response validation
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class Probability(BaseModel):
    species: str
    probability: float

class PredictionResponse(BaseModel):
    species: str
    species_id: int
    confidence: float
    probabilities: List[Probability]
    input_features: dict

class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    classes: List[str]
    accuracy: float

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Flower Classifier - FastAPI</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .form-group { margin: 15px 0; }
        label { display: inline-block; width: 150px; }
        input { padding: 5px; width: 200px; }
        button { background: #009688; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background: #00796B; }
        #result { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px; }
        .docs-link { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🌸 Iris Flower Classifier (FastAPI)</h1>
    <form id="predictionForm">
        <div class="form-group">
            <label>Sepal Length (cm):</label>
            <input type="number" step="0.1" name="sepal_length" value="5.1" required>
        </div>
        <div class="form-group">
            <label>Sepal Width (cm):</label>
            <input type="number" step="0.1" name="sepal_width" value="3.5" required>
        </div>
        <div class="form-group">
            <label>Petal Length (cm):</label>
            <input type="number" step="0.1" name="petal_length" value="1.4" required>
        </div>
        <div class="form-group">
            <label>Petal Width (cm):</label>
            <input type="number" step="0.1" name="petal_width" value="0.2" required>
        </div>
        <button type="submit">Predict Species</button>
    </form>
    <div id="result"></div>
    <div class="docs-link">
        <p><a href="/docs" target="_blank">📚 View API Documentation (Swagger UI)</a></p>
        <p><a href="/redoc" target="_blank">📖 View API Documentation (ReDoc)</a></p>
    </div>

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

@app.get("/", response_class=HTMLResponse)
async def home():
    """Render the home page with prediction form"""
    return HTML_TEMPLATE

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Predict iris species from flower measurements
    
    - **sepal_length**: Length of sepal in cm
    - **sepal_width**: Width of sepal in cm
    - **petal_length**: Length of petal in cm
    - **petal_width**: Width of petal in cm
    """
    try:
        # Prepare features array
        features_array = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        return PredictionResponse(
            species=target_names[prediction],
            species_id=int(prediction),
            confidence=float(probabilities[prediction]),
            probabilities=[
                Probability(species=target_names[i], probability=float(prob))
                for i, prob in enumerate(probabilities)
            ],
            input_features={
                "sepal_length": features.sepal_length,
                "sepal_width": features.sepal_width,
                "petal_length": features.petal_length,
                "petal_width": features.petal_width
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    """Get information about the trained model"""
    return ModelInfo(
        model_type="Logistic Regression",
        features=feature_names,
        classes=target_names,
        accuracy=float(model_data['accuracy'])
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    print("🌸 Starting FastAPI Iris Classifier API...")
    print("📍 API available at: http://localhost:8000")
    print("📝 Documentation:")
    print("   - Web Interface: http://localhost:8000/")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)
