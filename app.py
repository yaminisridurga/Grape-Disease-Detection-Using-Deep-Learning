from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import json

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("grape_disease_model.h5")

# Ensure 'static' directory exists for storing uploaded images
if not os.path.exists("static"):
    os.makedirs("static")

# Load class names correctly from a saved file (to match training order)
if os.path.exists("class_labels.json"):
    with open("class_labels.json", "r") as f:
        class_names = json.load(f)
else:
    class_names = ["Diseased", "Healthy"]  # Default order

print(f"Class Names Order: {class_names}")

def predict_image(image_path):
    """Preprocess the image and get model predictions."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Get index of highest probability
    confidence = np.max(prediction) * 100
    
    print(f"DEBUG: Raw Predictions: {prediction}")
    print(f"DEBUG: Predicted Class Index: {predicted_class}")
    print(f"DEBUG: Predicted Class Name: {class_names[predicted_class]}")
    
    return class_names[predicted_class], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", "uploaded.jpg")
            file.save(file_path)
            
            result, confidence = predict_image(file_path)
            return render_template("index.html", prediction=result, confidence=confidence, img_path=file_path)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
