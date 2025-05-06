# evaluate.py
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("Starting evaluation...")

# Check if test data exists
if not os.path.exists('X_test.npy') or not os.path.exists('y_test.npy'):
    print("Error: Test data files not found.")
    print("Please run model.py first to generate the test data.")
    exit(1)

def load_and_preprocess_test_data():
    try:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        
        # Ensure proper shape and normalization
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(-1, 64, 64, 1)  # Match IMG_SIZE from model.py
        X_test = X_test.astype('float32') / 255.0
        
        # Verify shapes match
        if len(X_test) != len(y_test):
            raise ValueError(f"Shape mismatch: X_test has {len(X_test)} samples but y_test has {len(y_test)} samples")
        
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load test data
    X_test, y_test = load_and_preprocess_test_data()
    if X_test is None or y_test is None:
        return

    print(f"Loaded test data shapes - X: {X_test.shape}, y: {y_test.shape}")

    # Load model with error handling
    try:
        print("Loading model...")
        model = load_model("nepali_sign_model.h5")
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get class labels
    dataset_dir = "sign_dataset"
    labels = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    # Evaluate model
    print("\nEvaluating model...")
    try:
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"\nOverall Metrics:")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Loss: {loss:.4f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run model.py first to generate the model and test data.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Get predictions and evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=labels))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, labels)
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()