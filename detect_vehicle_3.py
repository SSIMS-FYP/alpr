import numpy as np
from ultralytics import YOLO
import cv2
import easyocr
import os
import uuid
import requests

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# File paths
folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = 'license_plate_detector.pt'
VEHICLE_MODEL_WEIGHTS = 'yolov8n.pt'
CSV_DETECTIONS_PATH = './csv_detections/'

# Ensure the directory for CSV detections exists
if not os.path.exists(CSV_DETECTIONS_PATH):
    os.makedirs(CSV_DETECTIONS_PATH)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Vehicle classes in YOLOv8
vehicles = [2, 3, 5, 7]

# Load models
vehicle_detector = YOLO(VEHICLE_MODEL_WEIGHTS)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# API endpoint
API_ENDPOINT = "http://192.168.43.5:3001/vehicle/add/record"

vehicle_processed = False  # Flag to track if a vehicle has been processed

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    if not detections:
        return None, None  # Return None for both plate text and score if no detections

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    best_plate = None
    max_score = 0

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length * height / rectangle_size > 0.17:
            text, score = result[1].upper(), result[2]
            if score > max_score:
                max_score = score
                best_plate = text

    return best_plate, max_score  # Return the best plate text and its score

def detect_vehicle(img):
    vehicle_detections = vehicle_detector(img)[0]
    filtered_boxes = [box for box in vehicle_detections.boxes.data.tolist() if int(box[5]) in vehicles]
    return bool(filtered_boxes), filtered_boxes

def model_prediction(img):
    global vehicle_processed
    img_copy = img.copy()

    if not vehicle_processed:
        vehicle_detected, vehicle_boxes = detect_vehicle(img_copy)

        if vehicle_detected:
            for vehicle in vehicle_boxes:
                x1, y1, x2, y2, score, class_id = vehicle
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_detections = license_plate_detector(img_copy)[0]

            if license_detections.boxes.cls.tolist():
                best_plate_text = None
                max_plate_score = 0

                for license_plate in license_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)

                    if license_plate_text and license_plate_text_score > 0.8:
                        img_name = f'{uuid.uuid1()}.jpg'
                        cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
                        if license_plate_text_score > max_plate_score:
                            max_plate_score = license_plate_text_score
                            best_plate_text = license_plate_text

                if best_plate_text:
                    text_to_display = f"{best_plate_text} (Confidence: {max_plate_score:.2f})"
                    cv2.putText(img, text_to_display, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    send_data_to_api(best_plate_text)
                    vehicle_processed = True

    elif vehicle_processed:
        vehicle_detected, _ = detect_vehicle(img_copy)
        if not vehicle_detected:
            vehicle_processed = False

    return img

def send_data_to_api(plate_number):
    data = {"plateNumber": plate_number}
    response = requests.post(API_ENDPOINT, data)
    if response.status_code == 200:
        print(f"Successfully sent data: {data}")
    else:
        print(f"Failed to send data: {response.status_code}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = model_prediction(frame)
    cv2.imshow('License Plate Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
