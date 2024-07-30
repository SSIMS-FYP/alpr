import numpy as np
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import uuid
import os
import string
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
API_ENDPOINT = "http://192.168.1.33:3000/vehicle/add/record"

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    plate = []

    max_score = 0
    best_plate = None

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1].upper()
            scores += score
            plate.append((text, score))

            # Track the plate with highest confidence score
            if score > max_score:
                max_score = score
                best_plate = text

    if best_plate is not None:
        return best_plate, max_score
    else:
        return None, None

def detect_vehicle(img):
    vehicle_detections = vehicle_detector(img)[0]
    filtered_boxes = [box for box in vehicle_detections.boxes.data.tolist() if int(box[5]) in vehicles]
    if len(filtered_boxes) != 0:
        return True, filtered_boxes
    return False, []

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img_copy = img.copy()

    vehicle_detected, vehicle_boxes = detect_vehicle(img_copy)

    if vehicle_detected:
        for vehicle in vehicle_boxes:
            x1, y1, x2, y2, score, class_id = vehicle
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        license_detections = license_plate_detector(img_copy)[0]

        if len(license_detections.boxes.cls.tolist()) != 0:
            license_plate_crops_total = []
            max_plate_score = 0
            best_plate_text = None

            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
                img_name = '{}.jpg'.format(uuid.uuid1())
                cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

                if license_plate_text is not None and license_plate_text_score is not None:
                    if license_plate_text_score > 0.6:  # Check confidence score threshold
                        license_plate_crops_total.append(license_plate_crop)
                        results[license_numbers] = {}
                        results[license_numbers][license_numbers] = {'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                       'text': license_plate_text,
                                                                                       'bbox_score': score,
                                                                                       'text_score': license_plate_text_score}}
                        license_numbers += 1

                        # Track the best plate with highest confidence score
                        if license_plate_text_score > max_plate_score:
                            max_plate_score = license_plate_text_score
                            best_plate_text = license_plate_text

            # Draw text with confidence score on the image for the best plate
            if best_plate_text is not None:
                text_to_display = "{} (Confidence: {:.2f})".format(best_plate_text, max_plate_score)
                cv2.putText(img, text_to_display, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Send data to the API for the best plate
                send_data_to_api(best_plate_text)

            write_csv(results, os.path.join(CSV_DETECTIONS_PATH, 'detection_results.csv'))
            return img

    return img

def write_csv(results, filename):
    with open(filename, 'w') as f:
        for key, value in results.items():
            f.write("%s,%s\n" % (key, value))

def send_data_to_api(plate_number):
    data = {
        "plateNumber": plate_number
    }
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