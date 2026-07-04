from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("plant_classifier.keras")

class_names = [
    "Brown_Algae",
    "Green_Algae",
    "Red_Algae",
    "Seagrass"
]

frames_folder = "enhanced_frames_v2"

print("\n===== Classification Results =====\n")

for filename in os.listdir(frames_folder):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        filepath = os.path.join(frames_folder, filename)

        img = image.load_img(filepath, target_size=(224, 224))

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array / 255.0

        prediction = model.predict(img_array, verbose=0)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"Image: {filename}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 40)

print("\nClassification Completed!")              
