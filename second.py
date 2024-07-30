import numpy as np
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import uuid
import os

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# Directories
folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
csv_output_path = "./csv_detections/detection_results.csv"

# Ensure directories exist
os.makedirs(folder_path, exist_ok=True)
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

# Initialize EasyOCR reader and YOLO model
reader = easyocr.Reader(['en'], gpu=False)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# Set to keep track of already detected license plates
detected_plates = {}

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    if not detections:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    plate_texts = []
    total_score = 0

    for result in detections:
        bbox, text, score = result
        length = np.sum(np.subtract(bbox[1], bbox[0]))
        height = np.sum(np.subtract(bbox[2], bbox[1]))
        if length * height / rectangle_size > 0.17:
            text = text.upper()
            total_score += score
            plate_texts.append(text)

    if plate_texts:
        return " ".join(plate_texts), total_score / len(plate_texts)
    return "", 0

def model_prediction(img):
    img_copy = img.copy()
    license_detections = license_plate_detector(img_copy)[0]

    if license_detections.boxes.cls.tolist():
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)

            if license_plate_text and license_plate_text_score > 0.9:  # High confidence threshold
                if license_plate_text not in detected_plates or detected_plates[license_plate_text] < license_plate_text_score:
                    detected_plates[license_plate_text] = license_plate_text_score
                    img_name = '{}.jpg'.format(uuid.uuid1())
                    cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

                    # Draw text with confidence score on the image
                    text_to_display = f"{license_plate_text} (Confidence: {license_plate_text_score:.2f})"
                    cv2.putText(img, text_to_display, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        write_csv(detected_plates, csv_output_path, mode='w')  # Write to the CSV file

    return img

def write_csv(detected_plates, filename, mode='w'):
    data = [{"license_plate": plate, "confidence_score": score} for plate, score in detected_plates.items()]
    df = pd.DataFrame(data)
    df.to_csv(filename, mode=mode, index=False)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = model_prediction(frame)
    cv2.imshow('License Plate Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
