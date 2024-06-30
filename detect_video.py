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
license_plate_detector = YOLO('best.pt')

# Initialize EasyOCR reader.
# ['en'] specifies that we want to recognize text in English only.
reader = easyocr.Reader(['en'], gpu=False)

output_height = 720  # Desired height
output_width = 1280  # Desired width

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


def detect_vehicle_and_license_plate_in_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return

    # Get video frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Create the output folder for saving CSV and cropped grayscale license images, if it doesn't exist
    output_folder = 'output_results_videos'
    os.makedirs(output_folder, exist_ok=True)

    # Extract the base filename from the video path
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # Define the output video path
    output_video_path = os.path.join(output_folder, f'{video_filename}_video_output.avi')

    # Define the output CSV file path
    output_csv_path = os.path.join(output_folder, f'{video_filename}_results.csv')

    # Create the video writer for the output video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (output_width, output_height))

    # Open the CSV file for writing results
    save_results_to_csv({}, output_csv_path)  # Write an empty CSV file with header

    processed_car_ids = set()  # To keep track of processed car IDs

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for display
        frame = cv2.resize(frame, (output_width, output_height))

        # Detect vehicles using YOLOv8
        detections = coco_model(frame)[0]
        detections_ = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Filter by class IDs for vehicles (you can modify this based on your class IDs)
            if int(class_id) in [2, 3, 5, 7]:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates using YOLOv8
        license_plates = license_plate_detector(frame)[0]

        results = {}

        # Process license plates
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Get the corresponding vehicle information
            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

            if car_id in processed_car_ids:
                # Skip if the car ID has already been processed
                continue

            # Crop the license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

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

                    results[car_id] = {
                        'vehicle_bbox': [xcar1, ycar1, xcar2, ycar2],
                        'license_plate_bbox': [x1, y1, x2, y2],
                        'license_plate_text': formatted_license_plate_text,
                        'license_plate_text_score': license_plate_text_score
                    }

                    processed_car_ids.add(car_id)  # Add car ID to processed set

                    # Draw green bounding box for the license plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                    # Draw green bounding box for the vehicle
                    cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 4)

        # Add the frame with detections to the output video
        out.write(frame)

        # Save the results to the CSV file
        save_results_to_csv(results, output_csv_path)

        # Display the frame with detections
        cv2.imshow('Video with Detections', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to the input video
    input_video_path = 'sample.mp4'

    # Call the function to detect vehicles and license plates in the video
    detect_vehicle_and_license_plate_in_video(input_video_path)
