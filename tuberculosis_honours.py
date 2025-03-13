# -*- coding: utf-8 -*-
"""Tuberculosis Honours.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hX3CRhq1JaGaljbsyGfUgBGu3g-WNyNH
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D,
                                     GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics import roc_curve
from collections import Counter
import matplotlib.pyplot as plt

# Define the dataset directory
base_dir = '/content/drive/My Drive/Tuberculosis/TB_Chest_Radiography_Database/'  # Replace this with the actual path to your dataset

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16

# Data Augmentation and Generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Check class balance
print("Class distribution in training set:", Counter(train_generator.classes))
print("Class distribution in validation set:", Counter(validation_generator.classes))

# Define Edge Filter Block
def edge_filter_block(inputs):
    edge_filter = Conv2D(3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    return BatchNormalization()(edge_filter)

# Load Pre-trained Model (VGG16)
tf.keras.backend.clear_session()
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze the base model

# Input Layer
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Edge Filter Block
x = edge_filter_block(inputs)

# Pass through Pre-trained Model
x = base_model(x)

# Custom Convolutional Layers
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Global Average Pooling and Dense Layers
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

# Create Model
model = Model(inputs=inputs, outputs=outputs)

for layer in base_model.layers:
    layer.trainable = True  # Unfreeze all layers


# Compile the Model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
from tensorflow.keras.losses import BinaryFocalCrossentropy

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=BinaryFocalCrossentropy(gamma=2),  # Focuses on difficult examples
    metrics=['accuracy']
)


# Model Summary
model.summary()

# Train the Model
steps_per_epoch = min(np.sum(train_generator.classes == 0), np.sum(train_generator.classes == 1)) // BATCH_SIZE
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Save the Model
model.save("tb_classification_model.h5")

# Load the Model
model = load_model("tb_/classification_model.h5")

# Optimal Threshold Calculation
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
fpr, tpr, thresholds = roc_curve(y_true, y_pred[:len(y_true)])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
print(f"Optimal Threshold: {optimal_threshold}")

# Load and Predict on a Sample Image
sample_image_path = "/content/drive/My Drive/Tuberculosis/TB_Chest_Radiography_Database/Normal/Normal-11.png"  # Update this path to your sample image
# sample_image_path = "/content/drive/My Drive/Tuberculosis/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-1.png"
sample_image = image.load_img(sample_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
sample_image_array = image.img_to_array(sample_image) / 255.0  # Normalize the image
plt.imshow(sample_image_array)
plt.title("Sample Image for Prediction")
plt.axis("off")
plt.show()

# Preprocess the Sample Image
sample_image_resized = np.expand_dims(sample_image_array, axis=0)  # Add batch dimension

# Predict the Class
prediction = model.predict(sample_image_resized)
class_label = "Tuberculosis" if prediction[0] > optimal_threshold else "Normal"
print(f"Predicted Class: {class_label} (Confidence: {prediction[0][0]:.2f})")

import cv2
# Annotate the image if Tuberculosis is predicted
if class_label == "Tuberculosis":
    # Load the original image for annotation
    original_image = cv2.imread(sample_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Example bounding box coordinates (replace with your model's output if bounding boxes are predicted)
    bounding_boxes = [
        (50, 50, 200, 200),  # Example box 1 (x, y, width, height)
        (300, 100, 150, 150)  # Example box 2 (x, y, width, height)
    ]

    # Draw bounding boxes
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
        cv2.putText(original_image, 'Tuberculosis', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the annotated image
    plt.imshow(original_image)
    plt.title(f"Predicted Class: {class_label}")
    plt.axis("off")
    plt.show()
else:
    print("No bounding boxes to annotate. The prediction is Normal.")

# Calculate Risk Score
risk_score = prediction[0][0] * 100

# Categorize Risk Levels
if risk_score > 80:
    risk_level = "High Risk"
elif 50 <= risk_score <= 80:
    risk_level = "Moderate Risk"
else:
    risk_level = "Low Risk"

print(f"Risk Score: {risk_score:.2f}")
print(f"Risk Level: {risk_level}")

# Annotate Risk Score on Image if Tuberculosis is Predicted
if class_label == "Tuberculosis":
    for (x, y, w, h) in bounding_boxes:
        cv2.putText(original_image, f"Risk: {risk_score:.2f}%", (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green text

    # Display the annotated image
    plt.imshow(original_image)
    plt.title(f"Predicted Class: {class_label}, {risk_level}")
    plt.axis("off")
    plt.show()
else:
    print("Risk calculation not applicable as the prediction is 'Normal'.")

!pip install lime

from lime import lime_image
from skimage.segmentation import mark_boundaries

# Create a LIME explainer for images
explainer = lime_image.LimeImageExplainer()

# Function to predict probabilities for LIME
def predict_fn(images):
    # LIME expects a batch of images as input
    predictions = model.predict(images)
    # LIME requires the output to be 2D with probabilities for each class
    return np.hstack([1 - predictions, predictions])

# Explain the prediction for the sample image
explanation = explainer.explain_instance(
    sample_image_array,  # The image to explain
    predict_fn,          # Prediction function
    top_labels=1,        # Number of top labels to explain
    hide_color=0,        # Color to hide superpixels
    num_samples=1000     # Number of perturbed samples
)

# Visualize the explanation
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],  # Class to explain
    positive_only=True,              # Show only positive regions
    num_features=10,                 # Number of superpixels to highlight
    hide_rest=False                  # Show the rest of the image
)

# Display the LIME explanation
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(sample_image_array)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Explanation")
plt.axis("off")

plt.tight_layout()
plt.show()

