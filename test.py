import cv2 as cv
import numpy as np
import easyocr
from ultralytics import YOLO
import csv
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

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Initialize YOLO models for car detection and license plate detection
coco_model = YOLO('yolov8n.pt')  # YOLOv8 model for car detection
np_model = YOLO('best.pt')  # YOLOv8 model for license plate detection

# Define the list of vehicle class IDs from the COCO dataset (car, motorbike, truck)
vehicles = [2, 3, 5, 7]

def process_frame(frame, frame_number, results):
    # Use track() to identify instances and track them frame by frame
    detections = coco_model.track(frame, persist=True)[0]
    vehicle_bounding_boxes = []

    for detection in detections.boxes.data.tolist():
        if len(detection) < 7:
            continue
        x1, y1, x2, y2, _, score, class_id = detection
        # Check if the detected object is a vehicle and if its score is above the threshold
        if int(class_id) in vehicles and score > 0.5:
            vehicle_bounding_boxes.append([x1, y1, x2, y2])

    # Process each bounding box of detected vehicles
    for bbox in vehicle_bounding_boxes:
        x1, y1, x2, y2 = bbox
        roi = frame[int(y1):int(y2), int(x1):int(x2)]  # Region of interest (vehicle)

        # License plate detection for the region of interest
        license_plates = np_model(roi)[0]

        # Process each detected license plate
        for idx, license_plate in enumerate(license_plates.boxes.data.tolist()):
            if len(license_plate) < 6:
                continue
            plate_x1, plate_y1, plate_x2, plate_y2, _, _ = license_plate
            plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]  # Crop the license plate

            # Preprocess license plate images
            preprocess_and_save_images(plate, idx)

            # OCR
            np_text, np_score = read_license_plate(cv.cvtColor(plate, cv.COLOR_BGR2GRAY))  # Pass grayscale image
            if np_text is not None:
                # Draw boundary box around the vehicle and the license plate
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green for vehicle
                cv.rectangle(frame, (int(x1 + plate_x1), int(y1 + plate_y1)), (int(x1 + plate_x2), int(y1 + plate_y2)), (0, 0, 255), 2)  # Red for license plate
                cv.putText(frame, f'License Plate: {np_text}', (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv.putText(frame, f'Score: {np_score}', (int(x1), int(y1) - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                results.append([frame_number, x1, y1, x2, y2, plate_x1, plate_y1, plate_x2, plate_y2, np_text, np_score])

    return frame


# Preprocess license plate images
def preprocess_and_save_images(plate, idx):
    # De-colorize license plate image
    plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
    # Save the de-colorized license plate images
    cv.imwrite(f"plate_{idx}_gray.jpg", plate_gray)

# Function to read license plate using OCR
def read_license_plate(plate):
    detections = reader.readtext(plate)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        return text, score

    return None, None

# Main function to capture video from camera, process it in real-time, and write results to CSV
def capture_process_and_write_csv():
    results = []

    # Open the default camera (usually the primary laptop camera)
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Process the frame
        processed_frame = process_frame(frame, frame_number, results)

        # Display the frame with detections (optional, comment out if not needed)
        cv.imshow('Real-time Detections', processed_frame)

        # Break the loop if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()

    # Close all OpenCV windows
    cv.destroyAllWindows()

    # Write results to CSV file
    with open('detection_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Number', 'Car BBox X1', 'Car BBox Y1', 'Car BBox X2', 'Car BBox Y2',
                         'License Plate BBox X1', 'License Plate BBox Y1', 'License Plate BBox X2', 'License Plate BBox Y2',
                         'License Plate Number', 'License Plate Text Score'])
        for row in results:
            # Check if the confidence score is greater than 0.5 before writing to CSV
            if row[-1] > 0.5:
                writer.writerow(row)

if __name__ == "__main__":
    # Call the function to capture video from camera, process it in real-time, and write results to CSV
    capture_process_and_write_csv()
