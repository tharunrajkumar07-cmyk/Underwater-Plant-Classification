from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("resnet50_classifier.keras")

# Class names
class_names = [
    "Brown_Algae",
    "Green_Algae",
    "Red_Algae",
    "Seagrass"
]

# Load image
img = image.load_img("test.jpg", target_size=(224, 224))

# Convert image to array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

# Get predicted class
predicted_class = class_names[np.argmax(prediction)]

# Print output
print("Predicted Class:", predicted_class)
print("Confidence:", prediction)