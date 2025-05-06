import cv2
import os
import time
import sys
import shutil
import stat
import tempfile

def check_disk_space(directory, required_space_mb=500):
    """Check if there's enough disk space"""
    total, used, free = shutil.disk_usage(directory)
    free_mb = free // (2**20)  # Convert to MB
    return free_mb >= required_space_mb, free_mb

def check_permissions(directory):
    """Check if we have read/write permissions"""
    try:
        test_file = os.path.join(directory, 'test_permissions.txt')
        # Test write
        with open(test_file, 'w') as f:
            f.write('test')
        # Test read
        with open(test_file, 'r') as f:
            f.read()
        # Clean up
        os.remove(test_file)
        return True
    except (IOError, OSError) as e:
        print(f"Permission Error: {e}")
        return False

def ensure_directory_permissions(directory):
    """Ensure directory has proper permissions"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, mode=0o777, exist_ok=True)
        # Set full permissions
        os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        return True
    except Exception as e:
        print(f"Permission Error: {e}")
        return False

# Ensure the terminal supports Unicode
sys.stdout.reconfigure(encoding='utf-8')

# Get label from user and setup directory
label = input("Enter the character label (e.g. क, ख): ").strip()
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "sign_dataset")
save_dir = os.path.join(dataset_dir, label)

# Ensure both directories exist with proper permissions
print(f"Setting up directories...")
if not ensure_directory_permissions(dataset_dir):
    print("Failed to set up dataset directory")
    exit()
if not ensure_directory_permissions(save_dir):
    print("Failed to set up save directory")
    exit()

print(f"Saving images to: {save_dir}")

# Check disk space and permissions
has_space, free_mb = check_disk_space(save_dir)
if not has_space:
    print(f"Error: Not enough disk space. Only {free_mb}MB available.")
    exit()

if not check_permissions(save_dir):
    print(f"Error: Don't have proper permissions for directory {save_dir}")
    exit()

# Verify directory exists and is writable
if not os.path.exists(save_dir):
    print(f"Error: Directory {save_dir} could not be created.")
    exit()

# Add OpenCV version check and window creation with error handling
try:
    print(f"OpenCV version: {cv2.__version__}")
    cv2.namedWindow("Capture")  # Simplified window creation
except AttributeError:
    print("Error: OpenCV not properly installed. Please reinstall:")
    print("pip uninstall opencv-python")
    print("pip install --upgrade opencv-python")
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    cap.release()
    exit()

# Show first frame immediately
ret, frame = cap.read()
if ret:
    frame = cv2.flip(frame, 1)
    cv2.imshow("Capture", frame)
    cv2.waitKey(1)  # This is needed to actually display the window
    time.sleep(0.1) # Small delay to ensure window appears

start_count = len(os.listdir(save_dir))
img_count = 0
total_images = 200

print("Starting in 3 seconds... Get ready!")

# Show the camera feed and wait for 3 seconds
start_time = time.time()
while time.time() - start_time < 3:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Capture", frame)

    # Ensure the OpenCV window is responsive
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not ret:
    print("Exiting due to camera capture failure.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

SAVE_COLOR = True  # Set to True to save color images, False for grayscale

# Main capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)

    # Increased ROI size for better detail capture
    x1, y1, x2, y2 = 60, 60, 340, 340  # Larger ROI
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        print("Warning: ROI is empty, skipping frame.")
        continue

    if SAVE_COLOR:
        # Enhance color image
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        roi = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        resized = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE for better contrast in grayscale
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        resized = cv2.resize(enhanced, (128, 128), interpolation=cv2.INTER_LANCZOS4)

    # Save image to temp file first, then move to OneDrive folder
    if img_count < total_images:
        ext = ".png"
        filename = os.path.join(save_dir, f"{start_count + img_count}{ext}")
        filename = os.path.normpath(filename)
        
        try:
            # Convert image to bytes
            is_success, buffer = cv2.imencode(ext, resized)
            if is_success:
                # Write bytes directly to file
                with open(filename, "wb") as f:
                    f.write(buffer.tobytes())
                print(f"Saved image {img_count + 1}: {filename}")
                img_count += 1
            else:
                print(f"Failed to encode image to {filename}")
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            continue

    # Draw box and info
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"{label}: {img_count}/{total_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    if img_count >= total_images:
        cv2.putText(frame, "DONE!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= total_images:
        break

cap.release()
cv2.destroyAllWindows()
