from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, MaxPooling2D, Dropout,
                                   BatchNormalization, Input, GlobalAveragePooling2D,
                                   SeparableConv2D, Add, Activation)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
from collections import Counter
import gc

# Configuration
CONFIG = {
    'TRAIN_DATA': '/content/drive/My Drive/train_data/',
    'TEST_DATA': '/content/drive/My Drive/test_data/',
    'TRAIN_CSV': "/content/drive/My Drive/train_data/Train.csv",
    'TEST_CSV': "/content/drive/My Drive/test_data/Test.csv",
    'BATCH_SIZE': 16,  # Reduced for better gradient updates
    'IMG_SIZE': 256,
    'CHANNELS': 12,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.0001,
    'PATIENCE': 15,
    'VALIDATION_SPLIT': 0.2
}

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataProcessor:
    """Handles data loading and preprocessing with improved normalization"""

    @staticmethod
    def load_and_normalize_image(image_id, folder_path):
        """Enhanced image loading with better SAR processing"""
        try:
            image_path = os.path.join(folder_path, f"{image_id}.npy")
            if not os.path.exists(image_path):
                return None

            img = np.load(image_path)

            if len(img.shape) != 3 or img.shape[2] != 12:
                return None

            img_normalized = np.zeros_like(img, dtype=np.float32)

            # Optical bands (0-3) - Enhanced normalization
            for band in range(4):
                band_data = img[:, :, band].astype(np.float32)

                # Remove outliers using percentile clipping
                p2, p98 = np.percentile(band_data, [2, 98])
                band_data = np.clip(band_data, p2, p98)

                # Robust normalization
                if p98 > p2:
                    img_normalized[:, :, band] = (band_data - p2) / (p98 - p2)
                else:
                    img_normalized[:, :, band] = 0.5

            # SAR bands (4-11) - Improved SAR processing
            for band in range(4, 12):
                sar_data = img[:, :, band].astype(np.float32)

                # Convert to dB with proper handling
                sar_positive = np.abs(sar_data)
                sar_positive = np.maximum(sar_positive, 1e-12)
                sar_db = 10 * np.log10(sar_positive + 1e-12)

                # Clip extreme values
                sar_db = np.clip(sar_db, -50, 10)

                # Normalize to [0, 1]
                img_normalized[:, :, band] = (sar_db + 50) / 60

            return img_normalized

        except Exception as e:
            print(f"Error loading {image_id}: {str(e)}")
            return None

class BalancedDataGenerator(Sequence):
    """Improved data generator with better class balancing"""

    def __init__(self, image_ids, labels, folder_path, batch_size=16,
                 augment=False, shuffle=True, balance_classes=True):
        super().__init__()

        self.image_ids = np.array(image_ids)
        self.labels = np.array(labels)
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.balance_classes = balance_classes

        # Validate files
        self.valid_indices = self._validate_files()
        if len(self.valid_indices) == 0:
            raise FileNotFoundError(f"No valid files found in {folder_path}")

        # Create class-balanced indices if needed
        if self.balance_classes:
            self._create_balanced_indices()
        else:
            self.balanced_indices = self.valid_indices.copy()

        self.on_epoch_end()

        # Enhanced augmentation
        if self.augment:
            self.augmenter = ImageDataGenerator(
                rotation_range=45,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.2,
                zoom_range=0.3,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='reflect',
                brightness_range=[0.8, 1.2]
            )

    def _validate_files(self):
        """Validate file existence and readability"""
        valid_indices = []
        for idx, img_id in enumerate(self.image_ids):
            file_path = os.path.join(self.folder_path, f"{img_id}.npy")
            if os.path.exists(file_path):
                try:
                    test_img = np.load(file_path)
                    if len(test_img.shape) == 3 and test_img.shape[2] == 12:
                        valid_indices.append(idx)
                except:
                    continue
        return np.array(valid_indices)

    def _create_balanced_indices(self):
        """Create balanced sampling indices"""
        # Separate indices by class
        valid_labels = self.labels[self.valid_indices]
        class_0_indices = self.valid_indices[valid_labels == 0]
        class_1_indices = self.valid_indices[valid_labels == 1]

        # Oversample minority class
        min_class_size = min(len(class_0_indices), len(class_1_indices))
        max_class_size = max(len(class_0_indices), len(class_1_indices))

        # Create balanced dataset by oversampling
        if len(class_0_indices) < len(class_1_indices):
            # Oversample class 0
            oversample_indices = np.random.choice(class_0_indices,
                                                len(class_1_indices) - len(class_0_indices))
            class_0_indices = np.concatenate([class_0_indices, oversample_indices])
        else:
            # Oversample class 1
            oversample_indices = np.random.choice(class_1_indices,
                                                len(class_0_indices) - len(class_1_indices))
            class_1_indices = np.concatenate([class_1_indices, oversample_indices])

        self.balanced_indices = np.concatenate([class_0_indices, class_1_indices])
        print(f"Balanced dataset: {len(self.balanced_indices)} samples")

    def __len__(self):
        return int(np.ceil(len(self.balanced_indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.balanced_indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_images = []
        batch_labels = []

        for i in batch_indices:
            img_id = self.image_ids[i]
            label = self.labels[i]

            img = DataProcessor.load_and_normalize_image(img_id, self.folder_path)
            if img is not None:
                if self.augment:
                    img = self.augmenter.random_transform(img)
                batch_images.append(img)
                batch_labels.append(label)

        # Ensure we have a full batch
        while len(batch_images) < len(batch_indices) and len(batch_images) > 0:
            rand_idx = np.random.randint(0, len(batch_images))
            batch_images.append(batch_images[rand_idx])
            batch_labels.append(batch_labels[rand_idx])

        return np.array(batch_images), np.array(batch_labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.balanced_indices)

class ModelBuilder:
    """Builds improved CNN architecture"""

    @staticmethod
    def residual_block(x, filters, kernel_size=3):
        """Residual block for better gradient flow"""
        shortcut = x

        x = SeparableConv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = SeparableConv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)

        # Adjust shortcut if needed
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    @staticmethod
    def build_improved_model(input_shape):
        """Build improved CNN with residual connections"""
        inputs = Input(shape=input_shape)

        # Initial convolution
        x = Conv2D(32, (7, 7), strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

        # Residual blocks
        x = ModelBuilder.residual_block(x, 64)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = ModelBuilder.residual_block(x, 128)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = ModelBuilder.residual_block(x, 256)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)

        x = ModelBuilder.residual_block(x, 512)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        # Classification head
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        return model

# Custom metrics
def precision_m(y_true, y_pred):
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    true_positives = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    predicted_positives = tf.reduce_sum(y_pred)
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def recall_m(y_true, y_pred):
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    true_positives = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    possible_positives = tf.reduce_sum(y_true)
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Improved focal loss
def focal_loss(alpha=0.7, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        # Calculate focal loss
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = y_true * (1 - y_pred) ** gamma + (1 - y_true) * y_pred ** gamma

        # Binary crossentropy
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

        return tf.reduce_mean(alpha_factor * focal_weight * bce)

    return focal_loss_fixed

# Main execution
def main():
    print("Starting optimized landslide detection training...")

    # Load data
    train_df = pd.read_csv(CONFIG['TRAIN_CSV'])
    print(f"Loaded {len(train_df)} training samples")
    print(f"Class distribution: {train_df['label'].value_counts().to_dict()}")

    # Stratified split
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=CONFIG['VALIDATION_SPLIT'],
        random_state=42,
        stratify=train_df['label']
    )

    # Create generators
    train_gen = BalancedDataGenerator(
        image_ids=train_df['ID'].values[train_idx],
        labels=train_df['label'].values[train_idx],
        folder_path=CONFIG['TRAIN_DATA'],
        batch_size=CONFIG['BATCH_SIZE'],
        augment=True,
        balance_classes=True
    )

    val_gen = BalancedDataGenerator(
        image_ids=train_df['ID'].values[val_idx],
        labels=train_df['label'].values[val_idx],
        folder_path=CONFIG['TRAIN_DATA'],
        batch_size=CONFIG['BATCH_SIZE'],
        augment=False,
        balance_classes=False
    )

    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")

    # Build model
    input_shape = (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], CONFIG['CHANNELS'])
    model = ModelBuilder.build_improved_model(input_shape)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss=focal_loss(alpha=0.7, gamma=2.0),
        metrics=['accuracy', precision_m, recall_m, f1_m]
    )

    print("Model compiled successfully")
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_landslide_model.keras',
            monitor='val_f1_m',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_f1_m',
            mode='max',
            patience=CONFIG['PATIENCE'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Calculate class weights
    y_train = train_df['label'].values[train_idx]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")

    # Train model
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['EPOCHS'],
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Load best model
    model.load_weights('best_landslide_model.keras')

    # Threshold optimization
    print("\nOptimizing classification threshold...")
    val_predictions = []
    val_labels = []

    for i in range(len(val_gen)):
        batch_x, batch_y = val_gen[i]
        pred_batch = model.predict(batch_x, verbose=0)
        val_predictions.extend(pred_batch.flatten())
        val_labels.extend(batch_y.flatten())

    y_probs = np.array(val_predictions)
    y_true = np.array(val_labels)

    # Find optimal threshold using F1 score
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}

    for thresh in thresholds:
        y_pred = (y_probs > thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average='weighted')

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

            # Calculate per-class F1 scores
            report = classification_report(y_true, y_pred, output_dict=True)
            best_metrics = {
                'threshold': thresh,
                'weighted_f1': f1,
                'class_0_f1': report['0']['f1-score'],
                'class_1_f1': report['1']['f1-score']
            }

    print(f"\nBest threshold: {best_thresh:.3f}")
    print(f"Weighted F1: {best_metrics['weighted_f1']:.4f}")
    print(f"Class 0 F1: {best_metrics['class_0_f1']:.4f}")
    print(f"Class 1 F1: {best_metrics['class_1_f1']:.4f}")

    # Final evaluation
    y_pred_final = (y_probs > best_thresh).astype(int)
    print(f"\nFinal Classification Report:")
    print(classification_report(y_true, y_pred_final, target_names=['No Landslide', 'Landslide']))

    # Plot training history
    plot_training_history(history)

    # Generate test predictions
    generate_test_predictions(model, best_thresh)

    print("Training completed successfully!")
    return model, best_thresh

def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = ['accuracy', 'loss', 'f1_m', 'precision_m']
    titles = ['Accuracy', 'Loss', 'F1 Score', 'Precision']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = i // 2, i % 2

        if metric in history.history:
            axes[row, col].plot(history.history[metric], label=f'Training {title}')
            if f'val_{metric}' in history.history:
                axes[row, col].plot(history.history[f'val_{metric}'], label=f'Validation {title}')

            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(title)
            axes[row, col].legend()

    plt.tight_layout()
    plt.show()

def generate_test_predictions(model, threshold):
    """Generate test predictions"""
    test_df = pd.read_csv(CONFIG['TEST_CSV'])
    test_ids = test_df['ID'].values

    predictions = []
    batch_size = CONFIG['BATCH_SIZE']

    print(f"Generating predictions for {len(test_ids)} test images...")

    for i in range(0, len(test_ids), batch_size):
        batch_ids = test_ids[i:i+batch_size]
        batch_imgs = []

        for img_id in batch_ids:
            img = DataProcessor.load_and_normalize_image(img_id, CONFIG['TEST_DATA'])
            if img is not None:
                batch_imgs.append(img)
            else:
                batch_imgs.append(np.zeros((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], CONFIG['CHANNELS'])))

        batch_imgs = np.array(batch_imgs)
        probs = model.predict(batch_imgs, verbose=0).flatten()
        preds = (probs > threshold).astype(int)
        predictions.extend(preds)

        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {min(i+batch_size, len(test_ids))}/{len(test_ids)}")

    # Create submission
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'label': np.array(predictions, dtype=int)
    })

    submission_df.to_csv('optimized_submission.csv', index=False)
    print(f"Submission saved. Prediction distribution: {Counter(predictions)}")

if __name__ == "__main__":
    # Clear memory
    tf.keras.backend.clear_session()
    gc.collect()

    # Run main training
    model, best_threshold = main()
