import cv2
import numpy as np
import os
import uuid
import easyocr
from ultralytics import YOLO

# Initialize models and reader
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
coco_model = YOLO(COCO_MODEL_DIR)
reader = easyocr.Reader(['en'], gpu=False)

# Folder path to save cropped license plate images
folder_path = "./licenses_plates_imgs_detected/"

# Function to read license plate using EasyOCR
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    if not detections:
        return None
    plates = []
    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length * height / (license_plate_crop.shape[0] * license_plate_crop.shape[1]) > 0.17:
            plates.append(result[1].upper())
    return plates

# Function to perform model prediction on an image
def model_prediction(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    license_numbers = 0
    results = {}
    licenses_texts = []

    # Perform object detection using YOLO
    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    # Iterate over detected objects
    for detection in object_detections.boxes.data.tolist():
        xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
        if int(class_id) in [2, 3, 5, 7]:
            cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)

    # Iterate over detected license plates
    for license_plate in license_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
        img_name = '{}.jpg'.format(uuid.uuid1())
        cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        license_plate_texts = read_license_plate(license_plate_crop_gray)
        if license_plate_texts is not None:
            licenses_texts.extend(license_plate_texts)
            # Display extracted text on the image
            cv2.putText(img, ', '.join(license_plate_texts), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        results[license_numbers] = {
            'license_plate_bbox': [x1, y1, x2, y2],
            'license_plate_text': license_plate_texts
        }
        license_numbers += 1

    return img, licenses_texts, results

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    img, texts, detection_results = model_prediction(frame)

    # Display detection results
    cv2.imshow('License Plate Detection', img)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
