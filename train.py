import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset
DATASET_DIR = "sign_dataset"
IMG_SIZE = 64

X = []
y = []
labels = sorted(os.listdir(DATASET_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

# Load images
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
print(f"Looking for images in: {os.path.abspath(DATASET_DIR)}")

def load_image(path):
    """Load image with Unicode path support"""
    try:
        # Read file as bytes
        with open(path, 'rb') as f:
            img_bytes = bytearray(f.read())
        
        # Decode bytes to numpy array
        img_array = np.asarray(img_bytes, dtype=np.uint8)
        # Decode image
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            return img
            
        # Fallback to PIL if cv2 fails
        pil_img = Image.open(path).convert('L')
        return np.array(pil_img)
            
    except Exception as e:
        print(f"Failed to load image {path}: {str(e)}")
        return None

try: 
    raw_labels = [d for d in sorted(os.listdir(DATASET_DIR)) 
                  if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not raw_labels:
        print("Error: No valid folders found in dataset directory.")
        exit()
    print(f"Found labels: {raw_labels}")

    labels = []
    label_map = {}
    label_idx = 0

    for label in raw_labels:
        # Normalize path to handle Unicode
        folder = os.path.abspath(os.path.join(DATASET_DIR, label))
        folder = os.path.normpath(folder)
        
        if not os.path.exists(folder):
            print(f"Warning: Folder does not exist: {folder}")
            continue
            
        # Get files with Unicode path support
        try:
            files = [f for f in os.listdir(folder) 
                    if f.lower().endswith(valid_extensions)]
        except Exception as e:
            print(f"Error reading directory {folder}: {e}")
            continue
        
        print(f"\nProcessing '{label}': {len(files)} valid image files found")
        print(f"Folder path: {os.path.abspath(folder)}")
        valid_count = 0

        for filename in files:
            try:
                # Handle UTF-8 paths
                path = os.path.abspath(os.path.join(folder, filename))
                
                img = load_image(path)
                if img is None:
                    print(f"Error: Could not read image at: {path}")
                    print(f"Make sure the file is a valid image and has read permissions")
                    continue
                
                # Verify image dimensions
                print(f"Loading image: {path} - Original size: {img.shape}")
                if img.size == 0:
                    print(f"Error: Image is empty: {path}")
                    continue
                    
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                if img.shape != (IMG_SIZE, IMG_SIZE):
                    print(f"Error: Resize failed for: {path}")
                    continue
                    
                X.append(img)
                y.append(label_idx)
                valid_count += 1
                print(f"Successfully loaded image {valid_count}: {path}")

            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue

        if valid_count > 0:
            labels.append(label)
            label_map[label] = label_idx
            label_idx += 1
            print(f"Label '{label}': {valid_count} images loaded successfully")
        else:
            print(f"Warning: Label '{label}' has no valid images and will be skipped.")

except Exception as e:
    print(f"Fatal error: {str(e)}")
    exit()

if not X or not y:
    print("Error: No valid images found in dataset.")
    exit()

print(f"\nTotal images loaded: {len(X)}")
print(f"Number of classes: {len(labels)}")

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(np.array(y))

# After loading the data and before splitting, add class weights calculation
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y, axis=1)),
    y=np.argmax(y, axis=1)
)
class_weight_dict = dict(enumerate(class_weights))

print("\nClass weights to balance training:")
for label, weight in class_weight_dict.items():
    print(f"Class {labels[label]}: {weight:.2f}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for evaluation (save y_test before one-hot encoding)
print("Saving test data...")
np.save('X_test.npy', X_test)
np.save('y_test.npy', np.argmax(y_test, axis=1))  # Save original labels instead of one-hot encoded

# Before model creation, print number of classes
num_classes = len(labels)
print(f"Creating model with {num_classes} output classes")

# Create an improved CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # Explicitly use num_classes
])

# Compile with proper loss for multiclass
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model.summary()

# Train with early stopping and increased epochs
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,  # Increased patience for more epochs
    restore_best_weights=True,
    verbose=1
)

# Checkpoint to save best model
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Create data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Modify the model.fit call to use class weights and data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint],
    class_weight=class_weight_dict,
    verbose=1,
    steps_per_epoch=len(X_train) // 32
)

# Add training metrics visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Save final model
model.save("nepali_sign_model.h5")
