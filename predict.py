import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image, ImageDraw, ImageFont

# Constants
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for testing
ROI_SIZE = 280

def get_font():
    """Try to load a font that supports Devanagari"""
    font_paths = [
        "C:\\Windows\\Fonts\\Nirmala.ttf",  # Windows Devanagari font
        "C:\\Windows\\Fonts\\mangal.ttf",   # Alternative Windows font
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"  # Linux
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, 32)
    print("Warning: No Devanagari font found, using default font")
    return None

def draw_prediction(frame, text, position, color):
    """Draw text with Devanagari support"""
    font = get_font()
    if font:
        # Convert to PIL Image for proper Devanagari rendering
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text(position, text, font=font, fill=color)
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    else:
        # Fallback to OpenCV text
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def preprocess_frame(roi):
    try:
        # Match exactly the training preprocessing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE with same parameters as training
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Use same interpolation method as training
        resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        # Add debug visualization
        cv2.imshow('Preprocessed Input', resized)
        # Reshape and normalize exactly like training
        processed = resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
        return processed
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def detect_hand(roi):
    """Check if a hand is present in the ROI"""
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour is large enough to be a hand
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Adjust this threshold based on your ROI size
            return True
    return False

def verify_model_predictions(model, labels):
    print("\nVerifying model predictions...")
    # Create a random test input
    test_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1)
    
    # Get predictions
    predictions = model.predict(test_input, verbose=0)
    
    # Print probabilities for each class
    print("\nPrediction distribution:")
    for label, prob in zip(labels, predictions[0]):
        print(f"{label}: {prob:.4f}")
    
    # Check if predictions sum to approximately 1
    print(f"\nSum of probabilities: {np.sum(predictions[0]):.4f}")
    return np.sum(predictions[0]) > 0.99  # Should be close to 1

def normalize_predictions(predictions, temperature=1.0):
    """Apply temperature scaling to soften/sharpen predictions"""
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.exp(np.log(predictions) / temperature)
    return predictions / np.sum(predictions)

def main():
    try:
        # Load model with compile=False to avoid warning
        print("Loading model...")
        model = load_model("nepali_sign_model.h5", compile=False)
        
        # Add model summary
        print("\nModel architecture:")
        model.summary()
        
        # Compile with same parameters as training
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load labels and verify
        dataset_dir = "sign_dataset"
        labels = sorted([d for d in os.listdir(dataset_dir) 
                      if os.path.isdir(os.path.join(dataset_dir, d))])
        print(f"Loaded labels: {labels}")
        
        # Add this diagnostic code
        print("\nChecking dataset distribution:")
        for label in labels:
            path = os.path.join(dataset_dir, label)
            num_samples = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{label}: {num_samples} samples")
        
        if not labels:
            raise ValueError("No labels found in dataset directory")
            
        # Verify output layer matches number of classes
        num_classes = len(labels)
        output_shape = model.layers[-1].output.shape[-1]  # Changed from output_shape to output.shape
        print(f"\nNumber of classes in dataset: {num_classes}")
        print(f"Number of outputs in model: {output_shape}")
        
        if num_classes != output_shape:
            raise ValueError(f"Model output ({output_shape}) doesn't match number of classes ({num_classes})")
        
        if not verify_model_predictions(model, labels):
            print("Warning: Model predictions may not be properly normalized")
            
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Define ROI box in center of frame
        height, width = frame.shape[:2]
        x1 = width // 2 - ROI_SIZE // 2
        y1 = height // 2 - ROI_SIZE // 2
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE
        
        # Draw ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        roi = frame[y1:y2, x1:x2]
        if detect_hand(roi):
            processed = preprocess_frame(roi)
            if processed is not None:
                # Get prediction and normalize it
                prediction = model.predict(processed, verbose=0)
                # Apply softmax temperature scaling
                prediction[0] = normalize_predictions(prediction[0], temperature=0.5)  # Lower temperature for sharper predictions
                pred_idx = np.argmax(prediction)
                confidence = prediction[0][pred_idx]

                # Only show prediction if confidence is significantly higher than random
                min_confidence = 1.0 / len(labels)  # Random guess probability
                confidence_threshold = min_confidence * 2  # Must be at least 2x better than random

                if confidence > max(CONFIDENCE_THRESHOLD, confidence_threshold):
                    text = f"{labels[pred_idx]} ({confidence:.2f})"
                    frame = draw_prediction(frame, text, (x1, y1-35), (0, 255, 0))
        
        # Show preprocessed ROI
        cv2.imshow('Preprocessed', processed[0].reshape(IMG_SIZE, IMG_SIZE))
        
        cv2.imshow('Nepali Sign Language Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
