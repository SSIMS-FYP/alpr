import cv2
import numpy as np
from ultralytics import YOLO
import os
import hashlib

# Load YOLOv8 model for ID card detection
id_card_detector = YOLO('best.pt')

output_folder = 'license_images'
os.makedirs(output_folder, exist_ok=True)

# Maintain a set to store unique identifiers of detected ID cards
detected_id_cards = set()

# Function to calculate a unique identifier for an image
def calculate_image_hash(image):
    return hashlib.md5(image).hexdigest()

def detect_id_card_in_frame(frame):
    global detected_id_cards

    # Detect ID cards using YOLOv8
    id_cards = id_card_detector(frame)[0]

    for id_card in id_cards.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = id_card

        # Crop the ID card
        id_card_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # Calculate hash of the cropped ID card image
        image_hash = calculate_image_hash(id_card_crop.tobytes())

        # Check if the ID card already exists
        if image_hash in detected_id_cards:
            continue  # Skip saving if already detected

        # Save the cropped ID card image
        global frame_count
        image_name = f"id_card_{frame_count}.png"  # Unique name based on frame count
        cv2.imwrite(os.path.join(output_folder, image_name), id_card_crop)

        # Add the ID card's hash to the set of detected ID cards
        detected_id_cards.add(image_hash)

if __name__ == "__main__":
    # Open webcam
    cap = cv2.VideoCapture(0)

    frame_count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Detect ID cards in the frame and save them
        detect_id_card_in_frame(frame)

        # Increase frame count
        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
