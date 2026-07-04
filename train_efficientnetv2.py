import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast
)

# ====================================================
# Parameters
# ====================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ====================================================
# Load Dataset
# ====================================================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ====================================================
# Improve Dataset Performance
# ====================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# ====================================================
# Data Augmentation
# ====================================================

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.15),
    RandomZoom(0.15),
    RandomContrast(0.10)
], name="data_augmentation")

# ====================================================
# Build EfficientNetV2 (From Scratch)
# ====================================================
base_model = EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ====================================================
# Build Complete Model
# ====================================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x)

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

outputs = Dense(4, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

# ====================================================
# Compile Model
# ====================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ====================================================
# Display Model
# ====================================================

model.summary()

# ====================================================
# Train Model
# ====================================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# ====================================================
# Evaluate
# ====================================================

loss, accuracy = model.evaluate(val_dataset)

print(f"\nValidation Accuracy : {accuracy*100:.2f}%")

# ====================================================
# Save Model
# ====================================================

model.save("efficientnetv2_classifier.keras")

print("\nModel Saved Successfully!")
