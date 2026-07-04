import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = tf.keras.models.load_model("plant_classifier.h5")

# Class labels (same order as training folders)
class_names = ["Seagrass", "Green_Algae", "Brown_Algae", "Red_Algae"]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocessing (IMPORTANT)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    print("\nPredicted Class:", class_names[predicted_class])
    print("Confidence:", predictions)

predict_image("test.jpg")