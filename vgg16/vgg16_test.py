import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# SETUP
# ==========================================
tf.keras.utils.set_random_seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
TEST_DIR = os.path.join(BASE_DIR, 'test')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'vgg16_best.keras')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# LOAD TEST DATASET
# ==========================================
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

class_names = test_dataset.class_names
print(f"Class names (index order): {class_names}")
# Urutan alfabetis: 0 = NORMAL, 1 = PNEUMONIA

AUTOTUNE = tf.data.AUTOTUNE
test_dataset_eval = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# LOAD MODEL
# ==========================================
print(f"\nMemuat model dari: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# PREDIKSI
# ==========================================
print("\nMelakukan prediksi pada test set...")
y_true = np.concatenate([y.numpy() for _, y in test_dataset_eval], axis=0).flatten().astype(int)
y_prob = model.predict(test_dataset_eval).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# ==========================================
# METRIK
# ==========================================
acc = accuracy_score(y_true, y_pred)
# Pneumonia = kelas positif (label 1)
prec_pneumonia = precision_score(y_true, y_pred, pos_label=1)
rec_pneumonia = recall_score(y_true, y_pred, pos_label=1)
f1_pneumonia = f1_score(y_true, y_pred, pos_label=1)
cm = confusion_matrix(y_true, y_pred)

print("\n========== HASIL EVALUASI VGG16 ==========")
print(f"Accuracy              : {acc:.4f}")
print(f"Precision (Pneumonia) : {prec_pneumonia:.4f}")
print(f"Recall (Pneumonia)    : {rec_pneumonia:.4f}")
print(f"F1-Score (Pneumonia)  : {f1_pneumonia:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# ==========================================
# VISUALISASI CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=False
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - VGG16')
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'vgg16_confusion_matrix.png')
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"\nConfusion matrix disimpan di: {out_path}")
plt.show()
