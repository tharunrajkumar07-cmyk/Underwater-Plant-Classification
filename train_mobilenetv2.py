import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# Parameters
# ==========================================

DATASET_PATH = "Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20

# ==========================================
# Data Generator
# ==========================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# ==========================================
# Training Dataset
# ==========================================

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# ==========================================
# Validation Dataset
# ==========================================

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# Load MobileNetV2
# ==========================================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the base model
base_model.trainable = False

# ==========================================
# Build Model
# ==========================================

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# Model Summary
# ==========================================

model.summary()

print("\nTraining Started...\n")

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==========================================
# Print Final Accuracy
# ==========================================

print("\n===================================")
print(f"Final Training Accuracy   : {history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy : {history.history['val_accuracy'][-1]*100:.2f}%")
print("===================================\n")

# ==========================================
# Save Model
# ==========================================

model.save("plant_classifier.keras")

print("Training Completed!")
print("Model Saved as plant_classifier.keras")
