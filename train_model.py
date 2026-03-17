import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===============================
# 1️⃣ DATASET PATH
# ===============================
DATASET_DIR = "dataset"  
# dataset/
#   ├── downSyndrome/
#   └── healthy/

IMG_SIZE = 224
BATCH_SIZE = 16

# ===============================
# 2️⃣ DATA GENERATORS
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("✅ Class indices:", train_gen.class_indices)
# Expected:
# {'downSyndrome': 0, 'healthy': 1}

# ===============================
# 3️⃣ BUILD MODEL
# ===============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# ===============================
# 4️⃣ PHASE 1 – TRAIN CLASSIFIER HEAD
# ===============================
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("model", exist_ok=True)

checkpoint = ModelCheckpoint(
    "model/down_syndrome_modeltf29.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\n🚀 Phase 1: Training classifier head...\n")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)

# ===============================
# 5️⃣ PHASE 2 – FINE TUNING (CRITICAL FIX)
# ===============================
print("\n🔥 Phase 2: Fine-tuning last CNN layers...\n")

for layer in base_model.layers[-30:]:  # unfreeze last 30 layers
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # lower LR
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)

print("\n✅ Training completed successfully!")
print("📁 Model saved at: model/down_syndrome_modeltf29.h5")
