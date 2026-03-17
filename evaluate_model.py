import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
MODEL_PATH = "model/down_syndrome_modeltf29.h5"
IMG_SIZE = 224
BATCH_SIZE = 16

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- DATA GENERATOR (NO AUGMENTATION) ----------------
datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class indices:", test_gen.class_indices)
# {'downSyndrome': 0, 'healthy': 1}

# ---------------- PREDICTION ----------------
y_true = test_gen.classes
y_pred_prob = model.predict(test_gen).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

# ---------------- METRICS ----------------
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Down Syndrome", "Healthy"]
))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Down Syndrome", "Healthy"],
    yticklabels=["Down Syndrome", "Healthy"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
