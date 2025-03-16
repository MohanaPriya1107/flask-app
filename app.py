import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load TFLite model (use relative path)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best-fp16.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape dynamically
input_shape = input_details[0]['shape']  # (1, height, width, channels)

def preprocess_image(image):
    """ Preprocess the image to match model input. """
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize to match model input size
        img = cv2.resize(img, (input_shape[1], input_shape[2]))  
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0).astype(np.float32)  # Add batch dimension

        return img
    except Exception as e:
        return str(e)

def predict_freshness(image):
    """ Run the model and return the classification result. """
    try:
        img = preprocess_image(image)
        if isinstance(img, str):
            return f"Error in preprocessing: {img}"

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        # **Check model type**
        if len(pred.shape) == 2 and pred.shape[1] > 1:
            # Classification Model
            confidence = float(np.max(pred))  # Get confidence score
            return "Fresh" if confidence >= 0.7 else "Not Fresh"

        elif pred.shape == (1, 25200, 7):  # Object Detection Model (YOLO format)
            pred = pred[0]  # Remove batch dimension
            confidence_scores = pred[:, 4]  # Get confidence column
            threshold = 0.2  # Lower threshold to detect more objects

            valid_indices = np.where(confidence_scores > threshold)[0]
            if len(valid_indices) == 0:
                return "No Fish Detected"

            # Get best detection (highest confidence)
            best_idx = valid_indices[np.argmax(confidence_scores[valid_indices])]
            best_prediction = pred[best_idx]
            confidence = float(best_prediction[4])  # Confidence score

            # **Assign Labels Based on Confidence**
            return "Fresh" if confidence >= 0.7 else "Not Fresh"

        else:
            return "Unexpected Model Output Shape"
    except Exception as e:
        return f"Prediction Error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    try:
        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')

        # Run model inference
        result = predict_freshness(image)

        return jsonify({'label': result})  # Only returning "Fresh" or "Not Fresh"
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
