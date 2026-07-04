import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# =========================
# 1. Load model
# =========================
model = tf.keras.models.load_model("plant_classifier.keras")

# =========================
# 2. Class labels (EDIT THIS)
# =========================
class_names = ["Healthy", "Brown Spot", "Yellow Leaf", "White Disease"]

# =========================
# 3. Your image folder (CHANGE THIS)
# =========================
folder_path = "your_folder_name"   # 👈 PUT YOUR 89 IMAGES FOLDER NAME HERE

# =========================
# 4. Prediction list
# =========================
results = []

# =========================
# 5. Loop through images
# =========================
for img_name in os.listdir(folder_path):

    img_path = os.path.join(folder_path, img_name)

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        results.append([img_name, predicted_class, confidence])

        print(f"{img_name} ➜ {predicted_class} ({confidence:.2f}%)")

    except Exception as e:
        print(f"Error with {img_name}: {e}")

# =========================
# 6. Save results to CSV
# =========================
df = pd.DataFrame(results, columns=["Image", "Prediction", "Confidence"])
df.to_csv("prediction_results.csv", index=False)

print("\n✅ Done! Results saved as prediction_results.csv")