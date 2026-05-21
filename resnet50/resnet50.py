import os
import tensorflow as tf
from keras import layers, models, applications, callbacks
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

tf.keras.utils.set_random_seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 30
EPOCHS_STAGE_2 = 20

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

train_labels = np.concatenate([y.numpy() for _, y in train_dataset]).flatten().astype(int)
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight = dict(enumerate(weights))

data_augmentation = tf.keras.Sequential([
    layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)), # Scale tipis
    layers.RandomRotation(factor=0.05), # Rotate tipis
    layers.RandomBrightness(factor=0.1), # Ubah brightness
    layers.RandomContrast(factor=0.1), # Ubah contrast
    layers.GaussianNoise(stddev=0.1)
], name='data_augmentation')

def build_model():
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = applications.resnet.preprocess_input(x)

    base_model = applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet50_best.keras'),
        monitor='val_loss', 
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )
]

model, base_model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)
history_stage_1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE_1,
    callbacks=callbacks_list,
    class_weight=class_weight
)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)
history_stage_2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE_2,
    callbacks=callbacks_list,
    class_weight=class_weight
)