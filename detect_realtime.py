import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import easyocr
import csv
import util  # Import your existing util.py file
import os

# Initialize the SORT tracker
mot_tracker = Sort()

# Load YOLOv8 model for vehicle detection (coco_model) and license plate detection (license_plate_detector)
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detection_best.pt')

# Initialize EasyOCR reader.
# ['en'] specifies that we want to recognize text in English only.
reader = easyocr.Reader(['en'], gpu=False)

output_height = 320  # Desired height
output_width = 720  # Desired width

def save_results_to_csv(results, output_path):
    # Check if the file exists, if not, write the header
    write_header = not os.path.exists(output_path)

    with open(output_path, 'a', newline='') as csvfile:
        fieldnames = ['car_id', 'vehicle_bbox', 'license_plate_bbox', 'license_plate_text', 'license_plate_text_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is newly created
        if write_header:
            writer.writeheader()

        for car_id, data in results.items():
            writer.writerow({
                'car_id': car_id,
                'vehicle_bbox': data['vehicle_bbox'],
                'license_plate_bbox': data['license_plate_bbox'],
                'license_plate_text': data['license_plate_text'],
                'license_plate_text_score': data['license_plate_text_score']
            })

def detect_vehicle_and_license_plate_realtime():
    # Open the default camera (usually the primary laptop camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    # Create window to display video
    cv2.namedWindow("Real-time Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Detections", output_width, output_height)

    processed_car_ids = set()  # To keep track of processed car IDs

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for processing
        frame_resized = cv2.resize(frame, (output_width, output_height))

        # Detect vehicles using YOLOv8
        results = coco_model(frame_resized)

        if isinstance(results, list):
            continue  # No detections, continue to the next frame

        detections = results.xyxy[0]  # Accessing bounding boxes (x_min, y_min, x_max, y_max)

        detections_ = []

        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection.tolist()  # Convert tensor to list

            # You may want to filter by class_id here if needed

            detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates using YOLOv8
        license_plates = license_plate_detector(frame_resized)

        if isinstance(license_plates, list):
            continue  # No license plates detected, continue to the next frame

        license_plates = license_plates[0]

        results = {}

        # Process license plates
        for license_plate in license_plates.xywh:
            x1, y1, w, h = license_plate
            x2, y2 = x1 + w, y1 + h

            # Get the corresponding vehicle information
            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car([x1, y1, x2, y2], track_ids)

            if car_id in processed_car_ids:
                # Skip if the car ID has already been processed
                continue

            # Crop the license plate
            license_plate_crop = frame_resized[int(y1):int(y2), int(x1):int(x2), :]

            # Convert license plate image to grayscale
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate number
            license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_gray)

            if license_plate_text is not None:
                # Check if license plate complies with the format
                if util.license_complies_format(license_plate_text):
                    # Format the license plate text
                    formatted_license_plate_text = util.format_license(license_plate_text)

                    # Check if we already have a result for this car
                    if car_id not in results or results[car_id]['license_plate_text_score'] < license_plate_text_score:
                        results[car_id] = {
                            'vehicle_bbox': [xcar1, ycar1, xcar2, ycar2],
                            'license_plate_bbox': [x1, y1, x2, y2],
                            'license_plate_text': formatted_license_plate_text,
                            'license_plate_text_score': license_plate_text_score
                        }

                        # Draw green bounding box for the license plate
                        cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                        # Draw green bounding box for the vehicle
                        cv2.rectangle(frame_resized, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 4)

                    processed_car_ids.add(car_id)  # Add car ID to processed set

        # Display the frame with detections
        cv2.imshow('Real-time Detections', frame_resized)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Call the function to detect vehicles and license plates in real-time from laptop camera
    detect_vehicle_and_license_plate_realtime()
