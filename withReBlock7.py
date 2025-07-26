from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import datetime

# Define paths for the dataset
train_data = '/content/drive/My Drive/train_data/'
test_data = '/content/drive/My Drive/test_data/'
train_csv = "/content/drive/My Drive/train_data/Train.csv"
test_csv = "/content/drive/My Drive/test_data/Test.csv"

# Load Train.csv and inspect the data
train_df = pd.read_csv(train_csv)
print("Train.csv:")
print(train_df.head())

# Check distribution of labels
label_counts = train_df['label'].value_counts()
labels = ['No Landslide', 'Landslide']

plt.figure(figsize=(6, 4))
plt.bar(labels, label_counts.values, color=['skyblue', 'salmon'])
plt.xlabel("Class Label")
plt.ylabel("Frequency")
plt.title("Distribution of Labels in Training Set")
plt.show()

# Update paths
train_data_path = '/content/drive/My Drive/train_data/'
test_data_path = '/content/drive/My Drive/test_data/'

# Modified function to handle SAR data better with error handling
def load_and_normalize_npy_image(image_id, folder_path):
    try:
        image_path = os.path.join(folder_path, f"{image_id}.npy")
        img = np.load(image_path)

        # Ensure we have the expected shape
        if len(img.shape) != 3 or img.shape[2] != 12:
            print(f"Warning: Unexpected shape for {image_id}: {img.shape}")
            return None

        img_normalized = np.zeros_like(img, dtype=np.float32)

        # Optical bands (0-3)
        for band in range(4):
            band_data = img[:,:,band]
            if band_data.max() == band_data.min():
                img_normalized[:,:,band] = 0.5  # Set to middle value if constant
            else:
                img_normalized[:,:,band] = (band_data - band_data.min()) / \
                                         (band_data.max() - band_data.min())

        # SAR bands (4-11)
        for band in range(4, 12):
            sar_data = img[:,:,band]

            # Handle SAR data conversion
            sar_positive = np.abs(sar_data)
            # Add small epsilon to avoid log(0)
            sar_positive = np.maximum(sar_positive, 1e-10)
            img_dB = 10 * np.log10(sar_positive + 1e-6)

            # Normalize dB values
            if img_dB.max() == img_dB.min():
                img_normalized[:,:,band] = 0.5
            else:
                img_normalized[:,:,band] = (img_dB - img_dB.min()) / \
                                         (img_dB.max() - img_dB.min())

        return img_normalized

    except Exception as e:
        print(f"Error loading {image_id}: {str(e)}")
        return None

# Band descriptions
band_descriptions = [
    "Red (Optical)", "Green (Optical)", "Blue (Optical)", "Near Infrared (Optical)",
    "Descending VV (SAR)", "Descending VH (SAR)", "Descending Diff VV (SAR Change)",
    "Descending Diff VH (SAR Change)", "Ascending VV (SAR)", "Ascending VH (SAR)",
    "Ascending Diff VV (SAR Change)", "Ascending Diff VH (SAR Change)"
]

# Get sample images for visualization
try:
    available_ids = [f.split('.')[0] for f in os.listdir(train_data_path) if f.endswith('.npy')]
    if len(available_ids) >= 2:
        example_ids = np.random.choice(available_ids, 2, replace=False)

        for image_id in example_ids:
            img_normalized = load_and_normalize_npy_image(image_id, train_data_path)
            if img_normalized is not None:
                # Plot all 12 bands in a 3x4 grid
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))
                fig.suptitle(f"Sample Image ID: {image_id} - All 12 Bands", fontsize=16, y=1.02)

                for band in range(12):
                    row = band // 4
                    col = band % 4
                    axes[row, col].imshow(img_normalized[:, :, band], cmap='viridis')
                    axes[row, col].set_title(f"Band {band+1}: {band_descriptions[band]}", fontsize=10)
                    axes[row, col].axis('off')

                plt.tight_layout()
                plt.show()
except Exception as e:
    print(f"Error in visualization: {str(e)}")

# =====================================================
# IMPROVED DATA GENERATOR WITH ERROR HANDLING
# =====================================================

class LandslideDataGenerator(Sequence):
    def __init__(self, image_ids, labels, folder_path, batch_size=32, augment=False, shuffle=True):
        # Call parent constructor
        super().__init__()

        self.image_ids = image_ids
        self.labels = labels
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

        # Verify files exist before proceeding
        self.valid_indices = self._validate_files()
        if len(self.valid_indices) == 0:
            raise FileNotFoundError(f"No valid .npy files found in {folder_path}")

        print(f"Found {len(self.valid_indices)} valid files out of {len(self.image_ids)} total")
        self.on_epoch_end()

        if self.augment:
            self.augmenter = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='reflect'
            )

    def _validate_files(self):
        """Check which files actually exist and are readable"""
        valid_indices = []
        for idx, img_id in enumerate(self.image_ids):
            file_path = os.path.join(self.folder_path, f"{img_id}.npy")
            if os.path.exists(file_path):
                try:
                    # Try to load the file to make sure it's valid
                    test_img = np.load(file_path)
                    if len(test_img.shape) == 3 and test_img.shape[2] == 12:
                        valid_indices.append(idx)
                except Exception as e:
                    print(f"Skipping invalid file {img_id}: {str(e)}")
                    continue
        return valid_indices

    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.valid_indices[idx*self.batch_size : (idx+1)*self.batch_size]

        batch_images = []
        batch_labels = []

        for i in batch_indices:
            img_id = self.image_ids[i]
            label = self.labels[i]

            img = load_and_normalize_npy_image(img_id, self.folder_path)
            if img is not None:
                if self.augment:
                    # Apply augmentation
                    img = self.augmenter.random_transform(img)
                batch_images.append(img)
                batch_labels.append(label)
            else:
                # Skip invalid images
                continue

        # If we don't have enough valid images, pad with zeros
        while len(batch_images) < min(self.batch_size, len(batch_indices)):
            # Get a random valid image to duplicate
            if len(batch_images) > 0:
                rand_idx = np.random.randint(0, len(batch_images))
                batch_images.append(batch_images[rand_idx])
                batch_labels.append(batch_labels[rand_idx])
            else:
                # Last resort: create zero image
                batch_images.append(np.zeros((256, 256, 12), dtype=np.float32))
                batch_labels.append(0)

        return np.array(batch_images), np.array(batch_labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_indices)

# Create generators with validation
try:
    # Stratified split based on class distribution
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=42,
        stratify=train_df['label']
    )

    train_gen = LandslideDataGenerator(
        image_ids=train_df['ID'].values[train_idx],
        labels=train_df['label'].values[train_idx],
        folder_path=train_data_path,
        batch_size=32,
        augment=True
    )

    val_gen = LandslideDataGenerator(
        image_ids=train_df['ID'].values[val_idx],
        labels=train_df['label'].values[val_idx],
        folder_path=train_data_path,
        batch_size=32,
        augment=False
    )

    print(f"Successfully created generators")
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")

    # Test loading one batch
    test_batch, test_labels = train_gen[0]
    print(f"Batch shape: {test_batch.shape}")
    print(f"Labels shape: {test_labels.shape}")
    print(f"Unique labels: {np.unique(test_labels)}")

except Exception as e:
    print(f"Generator initialization failed: {str(e)}")
    raise

# Inspect batches
X_batch, y_batch = train_gen[0]
print("Train batch shape:", X_batch.shape)
print("Train labels shape:", y_batch.shape)
print("Train label distribution:", np.unique(y_batch, return_counts=True))

X_val_batch, y_val_batch = val_gen[0]
print("Validation batch shape:", X_val_batch.shape)
print("Validation labels shape:", y_val_batch.shape)
print("Validation label distribution:", np.unique(y_val_batch, return_counts=True))

# Define metrics
def precision_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Focal Loss Function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        return K.mean(-alpha * K.pow(1 - p_t, gamma) * K.log(p_t))
    return focal_loss_fixed

# Build the CNN Model
model = Sequential([
    Input(shape=X_batch.shape[1:]),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# Compile with Focal Loss and evaluation metrics
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.5),
    metrics=['accuracy', precision_m, recall_m, f1_m]
)

# Model summary
model.summary()

# Training setup
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "best_model_f1.keras"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_f1_m',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_f1_m',
    mode='max',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# Define class weights
class_weight = {0: 1., 1: 5.}

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[tensorboard_callback, checkpoint, early_stop],
    class_weight=class_weight,
    verbose=1
)

# Load best model
model.load_weights(checkpoint_path)

# ===============================
# Post-training threshold tuning
# ===============================
print("Collecting validation predictions for threshold tuning...")

# Collect all validation data manually to avoid iterator issues
val_predictions = []
val_labels = []

for i in range(len(val_gen)):
    batch_x, batch_y = val_gen[i]
    pred_batch = model.predict(batch_x, verbose=0)
    val_predictions.extend(pred_batch.flatten())
    val_labels.extend(batch_y.flatten())

y_probs = np.array(val_predictions)
y_true = np.array(val_labels)

print(f"Collected {len(y_true)} validation samples")
print(f"Label distribution: {np.unique(y_true, return_counts=True)}")

# Sweep through thresholds
thresholds = np.arange(0.1, 0.9, 0.01)
best_f1 = 0
best_thresh = 0.5

for t in thresholds:
    y_pred = (y_probs > t).astype(int)
    score = f1_score(y_true, y_pred, zero_division=0)
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print(f"\nBest F1 Score on Validation Set: {best_f1:.4f} at threshold {best_thresh:.2f}")

# Final evaluation with best threshold
y_pred_final = (y_probs > best_thresh).astype(int)
print(f"\nFinal Classification Report (threshold={best_thresh:.2f}):")
print(classification_report(y_true, y_pred_final, target_names=['No Landslide', 'Landslide']))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_final))

# Plot training history
if 'history' in locals():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot accuracy
    axes[0,0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0,0].set_title('Model Accuracy')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()

    # Plot loss
    axes[0,1].plot(history.history['loss'], label='Training Loss')
    axes[0,1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,1].set_title('Model Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()

    # Plot F1 score
    axes[1,0].plot(history.history['f1_m'], label='Training F1')
    axes[1,0].plot(history.history['val_f1_m'], label='Validation F1')
    axes[1,0].set_title('F1 Score')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('F1 Score')
    axes[1,0].legend()

    # Plot precision
    axes[1,1].plot(history.history['precision_m'], label='Training Precision')
    axes[1,1].plot(history.history['val_precision_m'], label='Validation Precision')
    axes[1,1].set_title('Precision')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Precision')
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

print("Training completed successfully!")
