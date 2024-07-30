import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import pandas as pd
import os
import uuid

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load YOLO model for license plate detection
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# Function to read license plate text using EasyOCR
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    if not detections:
        return None

    plate = []

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / (license_plate_crop.shape[0] * license_plate_crop.shape[1]) > 0.17:
            plate.append(result[1].upper())

    return " ".join(plate) if plate else None

# Function to perform license plate detection and recognition on an image
def detect_and_recognize_license_plate(image_path):
    img = cv2.imread(image_path)

    # Perform license plate detection
    license_detections = license_plate_detector(img)[0]

    if not len(license_detections.boxes.cls.tolist()):
        print("No license plate detected in the image.")
        return

    # Create a folder to save cropped license plate images
    folder_path = "./licenses_plates_imgs_detected/"
    os.makedirs(folder_path, exist_ok=True)

    # Process each detected license plate
    for idx, license_plate in enumerate(license_detections.boxes.data.tolist()):
        x1, y1, x2, y2, _, _ = license_plate

        # Crop the detected license plate region
        license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

        # Save the cropped license plate image
        img_name = f'{uuid.uuid1()}.jpg'
        cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

        # Read text from the cropped license plate image
        license_plate_text = read_license_plate(cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY))

        if license_plate_text:
            print(f"License Plate {idx + 1}: {license_plate_text}")
        else:
            print(f"No text detected on License Plate {idx + 1}")

# Path to the input image
image_path = "1.png"

# Detect and recognize license plates in the image
detect_and_recognize_license_plate(image_path)
