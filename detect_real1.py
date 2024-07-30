import numpy as np
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import uuid
from util import set_background, write_csv
import os
import string

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'

reader = easyocr.Reader(['en'], gpu=False)

vehicles = [2, 3, 5, 7]

license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]

    plate = []

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)

    if len(plate) != 0:
        return " ".join(plate), scores / len(plate)
    else:
        return " ".join(plate), 0

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img_copy = img.copy()

    license_detections = license_plate_detector(img_copy)[0]

    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())

            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None:
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}

                results[license_numbers][license_numbers] = {'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                               'text': license_plate_text,
                                                                               'bbox_score': score,
                                                                               'text_score': license_plate_text_score}}
                license_numbers += 1

                # Draw text with confidence score on the image
                text_to_display = f"{license_plate_text} (Confidence: {license_plate_text_score:.2f})"
                cv2.putText(img, text_to_display, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        write_csv(results, "./csv_detections/detection_results.csv")

        return img

    else:
        return img


def write_csv(results, filename):
    with open(filename, 'w') as f:
        for key, value in results.items():
            f.write("%s,%s\n" % (key, value))

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