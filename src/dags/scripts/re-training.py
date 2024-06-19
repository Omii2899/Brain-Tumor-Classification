import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from shutil import copy2

# Path to feedback directory and existing dataset
feedback_dir = "feedback_data"
existing_data_dir = "existing_data"

# Function to load existing data
def load_existing_data():
    data = []
    for file in os.listdir(existing_data_dir):
        if file.endswith(".json"):
            with open(os.path.join(existing_data_dir, file), 'r') as f:
                data.append(json.load(f))
    return data

# Function to load feedback data
def load_feedback_data():
    data = []
    for file in os.listdir(feedback_dir):
        if file.endswith(".json"):
            with open(os.path.join(feedback_dir, file), 'r') as f:
                data.append(json.load(f))
    return data

# Function to merge existing data with feedback
def merge_data(existing_data, feedback_data):
    existing_data_dict = {item['image_path']: item for item in existing_data}
    for feedback in feedback_data:
        image_path = feedback['image_path']
        if image_path in existing_data_dict:
            existing_data_dict[image_path]['label'] = feedback['label']
        else:
            existing_data_dict[image_path] = feedback
    return list(existing_data_dict.values())

# # Function to preprocess data
# def preprocess_data(data):
#     images = []
#     labels = []
#     for item in data:
#         image = tf.keras.preprocessing.image.load_img(item['image_path'], target_size=(224, 224))
#         image = tf.keras.preprocessing.image.img_to_array(image)
#         image = image / 255.0
#         images.append(image)
#         labels.append(1 if item['label'] == 'abnormal' else 0)
#     return np.array(images), np.array(labels)

# # Load existing data and feedback
existing_data = load_existing_data()
feedback_data = load_feedback_data()

# Merge data
merged_data = merge_data(existing_data, feedback_data)

# Preprocess data
images, labels = preprocess_data(merged_data)

# Split data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data generators
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# Load existing model
model = tf.keras.models.load_model('path_to_existing_model.h5')

# Retrain model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the retrained model
model.save('path_to_retrained_model.h5')
