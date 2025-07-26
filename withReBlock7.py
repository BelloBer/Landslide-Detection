from google.colab import drive
drive.mount('/content/drive')
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define paths for the dataset (remember to unzip the dataset first!)
#train_csv_path = 'data/Train.csv'  # Path to the training labels CSV file
#test_csv_path = 'data/Test.csv'    # Path to the test image IDs CSV file
#train_data_path = 'data/train_data'  # Folder where .npy train files are located
#test_data_path = 'data/test_data'    # Folder where .npy test files are located

#new ecode entry, trying to mount google drive

train_data = '/content/drive/My Drive/train_data1/'
test_data = '/content/drive/My Drive/test_data/'

train_csv = "/content/drive/My Drive/train_data1/Train.csv"
test_csv = "/content/drive/My Drive/test_data/Test.csv"


# Load Train.csv and inspect the data
train_df = pd.read_csv(train_csv)
print("Train.csv:")
print(train_df.head())

# Check distribution of labels
label_counts = train_df['label'].value_counts()
labels = ['No Landslide', 'Landslide']  # Map the labels 0 and 1 to descriptive names

plt.figure(figsize=(6, 4))
plt.bar(labels, label_counts.values, color=['skyblue', 'salmon'])
plt.xlabel("Class Label")
plt.ylabel("Frequency")
plt.title("Distribution of Labels in Training Set")
plt.show()

# Update these paths to match your Google Drive structure
train_data_path = '/content/drive/My Drive/train_data1/'  # Folder with .npy files
test_data_path = '/content/drive/My Drive/test_data/'    # Folder with test .npy files

# Modified function to handle SAR data better
def load_and_normalize_npy_image(image_id, folder_path):
    image_path = os.path.join(folder_path, f"{image_id}.npy")
    img = np.load(image_path)
    img_normalized = np.zeros_like(img, dtype=np.float32)

    # Optical bands (0-3)
    for band in range(4):
        img_normalized[:,:,band] = (img[:,:,band] - img[:,:,band].min()) / \
                                 (img[:,:,band].max() - img[:,:,band].min() + 1e-6)

    # SAR bands (4-11)
    for band in range(4, 12):
        # SAFE dB conversion
        sar_data = img[:,:,band]

        # Method 1: Absolute value (recommended)
        sar_positive = np.abs(sar_data)
        img_dB = 10 * np.log10(sar_positive + 1e-6)

        # Alternative if you know data type:
        # if amplitude data: sar_positive = sar_data**2
        # if intensity data: sar_positive = sar_data

        # Normalize dB values
        img_normalized[:,:,band] = (img_dB - img_dB.min()) / \
                                 (img_dB.max() - img_dB.min() + 1e-6)

    return img_normalized

# Band descriptions (updated with more details)
band_descriptions = [
    "Red (Optical)",
    "Green (Optical)",
    "Blue (Optical)",
    "Near Infrared (Optical)",
    "Descending VV (SAR)",
    "Descending VH (SAR)",
    "Descending Diff VV (SAR Change)",
    "Descending Diff VH (SAR Change)",
    "Ascending VV (SAR)",
    "Ascending VH (SAR)",
    "Ascending Diff VV (SAR Change)",
    "Ascending Diff VH (SAR Change)"
]

# Get 2 random samples (ensure they exist in the folder)
available_ids = [f.split('.')[0] for f in os.listdir(train_data_path) if f.endswith('.npy')]
example_ids = np.random.choice(available_ids, 2, replace=False)

for image_id in example_ids:
    try:
        img_normalized = load_and_normalize_npy_image(image_id, train_data_path)

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
        print(f"Could not process {image_id}: {str(e)}")
        continue

# =====================================================
# IMPROVED DATA GENERATOR WITH ERROR HANDLING
# =====================================================

class LandslideDataGenerator(Sequence):
    """
    Enhanced generator with:
    - File existence validation
    - Better error handling
    - Debugging support
    """
    
    def __init__(self, image_ids, labels, folder_path, batch_size=32, augment=False, shuffle=True):
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
                    with open(file_path, 'rb'):
                        valid_indices.append(idx)
                except:
                    continue
        return valid_indices

    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.valid_indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_ids = [self.image_ids[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        batch_images = []
        for image_id in batch_ids:
            try:
                img = load_and_normalize_npy_image(image_id, self.folder_path)
                if self.augment:
                    img = self.augmenter.random_transform(img)
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading {image_id}: {str(e)}")
                # Add zero image as fallback (adjust dimensions as needed)
                batch_images.append(np.zeros((256, 256, 12)))
                batch_labels.append(0)
                
        return np.array(batch_images), np.array(batch_labels)

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
    print("Please verify:")
    print(f"1. Path exists: {train_data_path}")
    print(f"2. Files exist: {len(os.listdir(train_data_path))} .npy files found")
    print(f"3. Sample file check: {os.path.exists(os.path.join(train_data_path, train_df['ID'].values[0] + '.npy'))}")
# Inspect one training batch
X_batch, y_batch = train_gen[0]
print("Train batch shape:", X_batch.shape)
print("Train labels shape:", y_batch.shape)
print("Train label distribution:", np.unique(y_batch, return_counts=True))

# Inspect one validation batch
X_val_batch, y_val_batch = val_gen[0]
print("Validation batch shape:", X_val_batch.shape)
print("Validation labels shape:", y_val_batch.shape)
print("Validation label distribution:", np.unique(y_val_batch, return_counts=True))

#Block6 - Start-
# Define the precision metric
def precision_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

# Recall metric
def recall_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# F1 Score metric
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Focal Loss Function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        return K.mean(-alpha * K.pow(1 - p_t, gamma) * K.log(p_t))
    return focal_loss_fixed

# Fetch a batch to define input shape
X_batch, _ = train_gen[0]

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
#Block6 - End -

#re-Block7
# Block 7: Train model and optimize F1 threshold
from sklearn.metrics import f1_score
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import datetime


# Define callbacks
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

# Define class weights (0 = no landslide, 1 = landslide)
class_weight = {0: 1., 1: 5.}  # Adjust as needed

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[tensorboard_callback, checkpoint, early_stop],
    class_weight=class_weight,
    verbose=1
)

# Load best model (just in case early_stop restored a different checkpoint)
model.load_weights(checkpoint_path)

# ===============================
# ✅ Post-training threshold tuning
# ===============================
# Predict probabilities on validation set
y_probs = model.predict(val_gen, verbose=1).flatten()

# Collect ground truth labels from val_gen
y_true = []
for _, y_batch in val_gen:
    y_true.extend(y_batch)
y_true = np.array(y_true).flatten()

# Sweep through thresholds
thresholds = np.arange(0.1, 0.9, 0.01)
best_f1 = 0
best_thresh = 0.5

for t in thresholds:
    y_pred = (y_probs > t).astype(int)
    score = f1_score(y_true, y_pred)
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print(f"\n Best F1 Score on Validation Set: {best_f1:.4f} at threshold {best_thresh:.2f}")
#re-block7 - end
