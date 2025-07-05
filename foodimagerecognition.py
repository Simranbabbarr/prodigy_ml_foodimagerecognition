import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Parameters
DATASET_PATH = "food"  # 🔁 changed from "food-101/images"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # You can increase this for better accuracy

# Calorie data (mock values; replace with real ones if available)
calorie_data = {
    "apple_pie": 296,
    "pizza": 285,
    "sushi": 200,
    "hamburger": 354,
    "french_fries": 365
}

# Step 1: Load images and labels
print("[INFO] Loading images...")
images = []
labels = []

for class_folder in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_folder)
    if not os.path.isdir(class_path):
        continue
    for image_file in os.listdir(class_path)[:150]:  # Limit to 150 per class for speed
        image_path = os.path.join(class_path, image_file)
        try:
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            image = preprocess_input(image)
            images.append(image)
            labels.append(class_folder)
        except Exception as e:
            print(f"[WARN] Could not load {image_path}: {e}")

images = np.array(images, dtype="float32")
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# Step 2: Build the model using MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(le.classes_), activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Step 3: Train the model
print("[INFO] Training model...")
H = model.fit(aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
              steps_per_epoch=len(X_train) // BATCH_SIZE,
              validation_data=(X_test, y_test),
              validation_steps=len(X_test) // BATCH_SIZE,
              epochs=EPOCHS)

# Step 4: Evaluate the model
print("[INFO] Evaluating model...")
predIdxs = model.predict(X_test, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)
trueIdxs = np.argmax(y_test, axis=1)
print("Classification Report:")
print(classification_report(trueIdxs, predIdxs, target_names=le.classes_))

# Step 5: Predict food and show calorie
def predict_food_and_calories(image_path):
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    class_idx = np.argmax(pred)
    class_label = le.classes_[class_idx]
    calories = calorie_data.get(class_label, "Unknown")
    print(f"Food: {class_label}, Estimated Calories: {calories}")
    return class_label, calories

# Example usage:
# predict_food_and_calories("path/to/image.jpg")

