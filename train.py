import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # silence oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'   # suppress TF info messages

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

print("TensorFlow version :", tf.__version__)
print("GPU available       :", tf.config.list_physical_devices('GPU'))
if not tf.config.list_physical_devices('GPU'):
    print("  Running on CPU — each epoch will take ~40-60 seconds")
    print("  To use GPU on Windows, downgrade: pip install tensorflow==2.10.0")


DATA_PATH         = "data"           # folder containing train/ val/ test/

# Exact subfolder names inside train/ val/ test/
CLASS_ACCIDENT    = "Accident"       # capital A — matches your dataset
CLASS_NO_ACCIDENT = "Non Accident"   # capital N, space — matches your dataset

IMG_SIZE          = (250, 250)       # must stay 250x250 for detection.py
BATCH_SIZE        = 32
EPOCHS_PHASE1     = 10               # frozen base — fast
EPOCHS_PHASE2     = 20               # fine-tune — slower but more accurate
LEARNING_RATE     = 0.0001
DROPOUT_RATE      = 0.5

# Output files — copy these into accident_detection_system/ when done
MODEL_JSON_OUT    = "model.json"
MODEL_WEIGHTS_OUT = "model_weights.weights.h5"
BEST_WEIGHTS_PATH = "best_weights.weights.h5"



print("\n" + "="*55)
print("  STEP 0 — Checking dataset folders")
print("="*55)

TRAIN_PATH = os.path.join(DATA_PATH, "train")
VAL_PATH   = os.path.join(DATA_PATH, "val")
TEST_PATH  = os.path.join(DATA_PATH, "test")

total_ok = True
for split_name, split_path in [("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)]:
    acc_path = os.path.join(split_path, CLASS_ACCIDENT)
    nac_path = os.path.join(split_path, CLASS_NO_ACCIDENT)

    acc_count = len([f for f in os.listdir(acc_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) \
                if os.path.exists(acc_path) else 0
    nac_count = len([f for f in os.listdir(nac_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) \
                if os.path.exists(nac_path) else 0

    ok = acc_count > 0 and nac_count > 0
    if not ok:
        total_ok = False
    status = "OK" if ok else "ERROR - missing or empty"
    print(f"  {split_name:6s}/  {CLASS_ACCIDENT}: {acc_count:5d}  |  {CLASS_NO_ACCIDENT}: {nac_count:5d}  [{status}]")

if not total_ok:
    print(f"\n  [ERROR] Some folders are missing or empty.")
    print(f"  Expected structure:")
    print(f"    {DATA_PATH}/train/{CLASS_ACCIDENT}/")
    print(f"    {DATA_PATH}/train/{CLASS_NO_ACCIDENT}/")
    print(f"    {DATA_PATH}/val/{CLASS_ACCIDENT}/")
    print(f"    {DATA_PATH}/val/{CLASS_NO_ACCIDENT}/")
    print(f"    {DATA_PATH}/test/{CLASS_ACCIDENT}/")
    print(f"    {DATA_PATH}/test/{CLASS_NO_ACCIDENT}/")
    raise SystemExit(1)

print("\n  All folders found!")

# ============================================================
#  STEP 1 — Data generators with augmentation
# ============================================================

print("\n" + "="*55)
print("  STEP 1 — Creating data generators")
print("="*55)

# Heavy augmentation on training — simulates real-world variation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],   # handles day/night/glare
    shear_range=0.1,
    fill_mode='nearest'
)

# No augmentation on val/test — only rescale
eval_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = eval_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_data = eval_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\n  Class mapping : {train_data.class_indices}")
print(f"  Train samples : {train_data.samples}")
print(f"  Val samples   : {val_data.samples}")
print(f"  Test samples  : {test_data.samples}")

# FIXED: correct check — 'Accident' with capital A
if train_data.class_indices.get(CLASS_ACCIDENT) == 0:
    print(f"\n  Class order OK — '{CLASS_ACCIDENT}' is index 0 (matches detection.py)")
else:
    print(f"\n  [WARNING] '{CLASS_ACCIDENT}' is not index 0!")
    print(f"  detection.py expects Accident at index 0.")
    print(f"  Current mapping: {train_data.class_indices}")

# ============================================================
#  STEP 2 — Class weights to fix imbalance
# ============================================================

print("\n" + "="*55)
print("  STEP 2 — Computing class weights")
print("="*55)

labels  = train_data.classes
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(weights))
print(f"  Class weights : {class_weight_dict}")
print("  (Higher weight = model pays more attention to that class)")

# ============================================================
#  STEP 3 — Build MobileNetV2 model
# ============================================================

print("\n" + "="*55)
print("  STEP 3 — Building MobileNetV2 model")
print("="*55)

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'          # pretrained on 1.4M images
)
base_model.trainable = False    # freeze in phase 1

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # Accident, Non Accident
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Total params     : {model.count_params():,}")
trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"  Trainable params : {trainable:,}  (base frozen)")

# ============================================================
#  STEP 4 — Phase 1: Train top layers only
# ============================================================

print("\n" + "="*55)
print(f"  STEP 4 — Phase 1: Top layers ({EPOCHS_PHASE1} epochs)")
print("="*55)

callbacks_p1 = [
    # FIXED: .weights.h5 extension required by Keras 3
    ModelCheckpoint(
        BEST_WEIGHTS_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weight_dict,
    callbacks=callbacks_p1,
    verbose=1
)

best_p1 = max(history1.history['val_accuracy'])
print(f"\n  Phase 1 best val accuracy: {best_p1*100:.2f}%")

# ============================================================
#  STEP 5 — Phase 2: Fine-tune entire model
# ============================================================

print("\n" + "="*55)
print(f"  STEP 5 — Phase 2: Fine-tuning ({EPOCHS_PHASE2} epochs)")
print("="*55)

# Unfreeze top 50 layers of MobileNetV2
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile with 10x lower LR to avoid destroying pretrained weights
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

trainable_now = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"  Trainable params now : {trainable_now:,}  (top 50 base layers unfrozen)")

callbacks_p2 = [
    # FIXED: .weights.h5 extension
    ModelCheckpoint(
        BEST_WEIGHTS_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
    initial_epoch=len(history1.history['loss']),
    class_weight=class_weight_dict,
    callbacks=callbacks_p2,
    verbose=1
)

best_p2 = max(history2.history['val_accuracy'])
print(f"\n  Phase 2 best val accuracy: {best_p2*100:.2f}%")

# ============================================================
#  STEP 6 — Evaluate on test set
# ============================================================

print("\n" + "="*55)
print("  STEP 6 — Test set evaluation")
print("="*55)

model.load_weights(BEST_WEIGHTS_PATH)
test_loss, test_acc = model.evaluate(test_data, verbose=1)

print(f"\n  Test accuracy : {test_acc * 100:.2f}%")
print(f"  Test loss     : {test_loss:.4f}")

if test_acc >= 0.92:
    print("  Result : Excellent! Ready for deployment.")
elif test_acc >= 0.82:
    print("  Result : Good. Add more images to push higher.")
elif test_acc >= 0.70:
    print("  Result : Fair. Collect more real CCTV footage frames.")
else:
    print("  Result : Low accuracy. Your dataset is small (791 images).")
    print("           Download more data from Kaggle to improve accuracy.")

# ============================================================
#  STEP 7 — Save model (drop-in for detection.py)
# ============================================================

print("\n" + "="*55)
print("  STEP 7 — Saving model files")
print("="*55)

model_json = model.to_json()
with open(MODEL_JSON_OUT, "w") as f:
    f.write(model_json)

model.save_weights(MODEL_WEIGHTS_OUT)

print(f"  Saved : {MODEL_JSON_OUT}")
print(f"  Saved : {MODEL_WEIGHTS_OUT}")
print()
print("  Next steps:")
print(f"  1. Copy {MODEL_JSON_OUT}        → accident_detection_system/")
print(f"  2. Copy {MODEL_WEIGHTS_OUT}  → accident_detection_system/")
print("  3. Run: python main.py")

# ============================================================
#  STEP 8 — Save training plot
# ============================================================

print("\n" + "="*55)
print("  STEP 8 — Saving training plot")
print("="*55)

acc      = history1.history['accuracy']     + history2.history['accuracy']
val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss     = history1.history['loss']         + history2.history['loss']
val_loss = history1.history['val_loss']     + history2.history['val_loss']
epochs_x = range(1, len(acc) + 1)
split    = len(history1.history['accuracy'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_x, acc,     'b-o', label='Train',      markersize=4)
ax1.plot(epochs_x, val_acc, 'r-o', label='Validation', markersize=4)
ax1.axvline(x=split, color='gray', linestyle='--', label='Fine-tune start')
ax1.set_title('Accuracy per epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_x, loss,     'b-o', label='Train',      markersize=4)
ax2.plot(epochs_x, val_loss, 'r-o', label='Validation', markersize=4)
ax2.axvline(x=split, color='gray', linestyle='--', label='Fine-tune start')
ax2.set_title('Loss per epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Training Results  |  Test Accuracy: {test_acc*100:.2f}%', fontsize=13)
plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("  Plot saved : training_history.png")

print("\n" + "="*55)
print("  TRAINING COMPLETE")
print("="*55)