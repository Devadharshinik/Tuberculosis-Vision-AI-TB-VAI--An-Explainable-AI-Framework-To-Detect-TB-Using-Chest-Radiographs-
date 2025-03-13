from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import io
import base64

app = Flask(__name__)

# Load the trained model
model = load_model("classification_model.h5")
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict and generate results
def predict_and_explain(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path)
    img_array = preprocess_image(img)

    # Model prediction
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    class_label = "Tuberculosis" if confidence > 0.5 else "Normal"

    # Risk score calculation
    risk_score = confidence * 100
    if risk_score > 80:
        risk_level = "High Risk"
    elif 50 <= risk_score <= 80:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # LIME explanation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array[0],  # The image to explain
        lambda x: np.hstack([1 - model.predict(x), model.predict(x)]),  # Prediction function
        top_labels=1,
        hide_color=0,
        num_samples=100
    )
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Convert LIME explanation to a displayable image
    lime_img = mark_boundaries(temp, mask)
    lime_img_path = "lime_explanation.png"
    plt.imsave(lime_img_path, lime_img)

    return class_label, confidence, risk_score, risk_level, lime_img_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    # Save the uploaded image
    img_file = request.files['image']
    img_path = os.path.join('uploads', img_file.filename)
    os.makedirs('uploads', exist_ok=True)
    img_file.save(img_path)

    # Perform prediction and explanation
    class_label, confidence, risk_score, risk_level, lime_img_path = predict_and_explain(img_path)

    # Convert LIME explanation to base64 for rendering in HTML
    with open(lime_img_path, "rb") as lime_file:
        lime_base64 = base64.b64encode(lime_file.read()).decode('utf-8')

    # Return results
    return render_template('result.html', 
                           class_label=class_label, 
                           confidence=confidence, 
                           risk_score=risk_score, 
                           risk_level=risk_level, 
                           lime_img=lime_base64)

if __name__ == '__main__':
    app.run(debug=True)
