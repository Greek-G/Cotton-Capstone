from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load your trained model
model = load_model("cotton_disease_model_fixed.keras")
print("✅ Model loaded successfully!")

# Define class names (make sure these are in same order as training)
CLASS_NAMES = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

@app.route('/')
def home():
    return "Cotton Disease Detection API is Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image file exists in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image directly from memory without saving
        img = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize image
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        print(f"🖼️ Processing image: {file.filename}")
        print(f"📐 Image shape: {img_array.shape}")

        # Make prediction
        predictions = model.predict(img_array)
        
        # Convert predictions to native Python types
        predictions_native = convert_to_serializable(predictions[0])
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))  # Convert to native float

        print(f"🎯 Prediction: {predicted_class} ({confidence*100:.2f}%)")

        # Prepare all predictions
        all_predictions = {}
        for i, class_name in enumerate(CLASS_NAMES):
            all_predictions[class_name] = round(float(predictions[0][i]) * 100, 2)

        response = {
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'all_predictions': all_predictions
        }

        # Ensure everything is serializable
        response = convert_to_serializable(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'classes': CLASS_NAMES
    })

if __name__ == '__main__':
    print("🚀 Starting Cotton Disease Detection Server...")
    print("📊 Model Info:")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Classes: {CLASS_NAMES}")
    app.run(debug=True, host='0.0.0.0', port=5000)