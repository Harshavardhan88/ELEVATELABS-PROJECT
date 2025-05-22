# collect_data.py
import cv2
import os
import time

# -----------------------------
# SET YOUR LABELS HERE
labels = ['A', 'B', 'C','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Change or expand as needed
dataset_path = 'dataset_custom'
num_images = 2000  # Number of images per label
delay = 5         # Delay before capturing starts
capture_delay = 0.05  # Time between captures
# -----------------------------

# Create folders if they don't exist
for label in labels:
    label_dir = os.path.join(dataset_path, label)
    os.makedirs(label_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

for label in labels:
    print(f"\n📸 Get ready to capture for label: {label}")
    time.sleep(delay)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip and draw info
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Label: {label} ({count}/{num_images})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture Sign Language Data", frame)

        # Save frame
        img_name = os.path.join(dataset_path, label, f"{label}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1
        time.sleep(capture_delay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("✅ Data collection completed.")
cap.release()
cv2.destroyAllWindows()
