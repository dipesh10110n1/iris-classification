"""
train_and_save_model.py
Trains the iris classifier and saves it as a pickle file
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Load and prepare data
iris = load_iris()

X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=50, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("MODEL TRAINING COMPLETED")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Save model and scaler
model_data = {
    'model': model,
    'scaler': scaler,
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist(),
    'accuracy': accuracy
}

with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Model saved as 'iris_model.pkl'")

# Test loading
with open('iris_model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
    print(f"✓ Model loaded successfully")
    print(f"  Features: {loaded_data['feature_names']}")
    print(f"  Classes: {loaded_data['target_names']}")
