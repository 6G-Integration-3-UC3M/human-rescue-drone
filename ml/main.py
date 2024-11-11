from ultralytics import YOLO
import torch
import cv2
import requests

from api import get_drone_rules, get_description_from_image_with_groq_cloud

# Define some constants
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for initial filtering
GREEN = (0, 255, 0)
DEVICE = "pc"  # Use 'jetson' for the RICOH THETA X with Jetson Nano, or 'pc' for a regular webcam
NUM_FOLDS = 1  # Change this value to increase the number of folds for division (e.g., 2, 4, 8)
ML_MODEL = "yolo11n.pt"


# Server configuration
URL_SERVER = "http://localhost:3000"
DRONE_IP = "12.12.12.13"
DRONE_SECRET = "DONT_SHARE_THIS_SECRET"
MISSION_NAME = "Apollo Mission 1"

# Get detection rules from the server
rules = get_drone_rules(URL_SERVER, DRONE_IP, DRONE_SECRET, MISSION_NAME)
print(f"Rules: {rules}")

# Initialize video capture based on the device
if DEVICE == "jetson":
    cap = cv2.VideoCapture(
        "thetauvcsrc mode=4K ! queue! h264parse! nvv4l2decoder ! queue ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink"
    )
else:
    cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Load YOLOv8 model and move it to the GPU if using Jetson
if DEVICE == "jetson":
    model = YOLO(ML_MODEL).to("cuda")  # Ensure the model runs on CUDA
else:
    model = YOLO(ML_MODEL)

# Get class names from the model
class_names = model.names  # Retrieve class names

while True:
    ret, frame = cap.read()

    # Check if frame is successfully read
    if not ret:
        print("Failed to grab frame")
        break

    # Get the original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Calculate dimensions for each fold
    fold_height = original_height // NUM_FOLDS
    fold_width = original_width // NUM_FOLDS

    for i in range(NUM_FOLDS):
        for j in range(NUM_FOLDS):
            # Alternate between vertical and horizontal splits
            if (i + j) % 2 == 0:  # Vertical split
                x_start = j * fold_width
                y_start = 0
                x_end = (j + 1) * fold_width if j + 1 < NUM_FOLDS else original_width
                y_end = original_height
            else:  # Horizontal split
                x_start = 0
                y_start = i * fold_height
                x_end = original_width
                y_end = (i + 1) * fold_height if i + 1 < NUM_FOLDS else original_height

            # Extract the fold from the frame
            fold = frame[y_start:y_end, x_start:x_end]

            # Calculate the padding needed to make the fold height and width divisible by 32
            pad_height = (32 - fold.shape[0] % 32) % 32
            pad_width = (32 - fold.shape[1] % 32) % 32

            # Pad the fold to make it divisible by 32
            pad_color = (0, 0, 0)  # Padding color (black)
            fold_padded = cv2.copyMakeBorder(fold, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=pad_color)

            # Convert the fold to a tensor
            fold_tensor = torch.tensor(fold_padded, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            # Move the fold tensor to GPU if using Jetson
            if DEVICE == "jetson":
                fold_tensor = fold_tensor.to("cuda")

            # Run the YOLO model on the fold (perform inference)
            detections = model(fold_tensor)[0]

            # Scale factors for bounding box adjustments
            x_scale = (x_end - x_start) / original_width
            y_scale = (y_end - y_start) / original_height

            for data in detections.boxes.data.tolist():
                # Extract the confidence associated with the detection
                confidence = data[4]

                # Get the class ID and label
                class_id = int(data[5])  # The class ID is stored in the 5th index
                label = class_names[class_id]  # Get the label from class names

                # Check against rules
                for rule_id, rule in rules.items():
                    # Ensure the rule is checking for a person with sufficient confidence
                    if label == rule['object'] and confidence >= rule['confidence']['value']:
                    # if True:
                        # Draw the bounding box on the frame if conditions are met
                        xmin, ymin, xmax, ymax = (
                            int(data[0] * x_scale) + x_start,
                            int(data[1] * y_scale) + y_start,
                            int(data[2] * x_scale) + x_start,
                            int(data[3] * y_scale) + y_start,
                        )
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)

                        # Prepare the text for the label and confidence
                        text = f"{label}: {confidence:.2f}"
                        # Display the label above the bounding box
                        cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

                        image_encoded = cv2.imencode('.jpg', frame)[1].tobytes()  # Encode the image to send

                        files = {
                            "image": ('detected_image.jpg', image_encoded, 'image/jpeg')  # Provide a filename and content type
                        }

                        description = get_description_from_image_with_groq_cloud(image_encoded)

                        detection_data = {
                            "droneIp": DRONE_IP,
                            "secret": DRONE_SECRET,
                            "missionName": MISSION_NAME,
                            "detectedObject": label,
                            "confidence": confidence,
                            "description": description,
                        }

                        print(f"Sending detection data: {detection_data}")

                        # Send the POST request with data and files
                        response = requests.post(f"{URL_SERVER}/api/detection/add", data=detection_data, files=files)

                        if response.status_code == 200:
                            print("Response from server:", response.json())
                        else:
                            print(f"Failed to get a valid response. {response.text}")

                        break  # Exit the loop once a matching rule is found

    # Display the frame with detections
    cv2.imshow("Input", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
