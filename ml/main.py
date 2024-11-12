import threading
import time
from ultralytics import YOLO
import torch
import cv2
import requests
from api import get_drone_rules, get_description_from_image_with_groq_cloud
from collections import deque
import aiohttp
import asyncio

# Constants
GREEN = (0, 255, 0)
DEVICE = "pc"  # Use 'jetson' for the RICOH THETA X with Jetson Nano, or 'pc' for a regular webcam
NUM_FOLDS = 1  # Number of folds for division
ML_MODEL = "yolo11n.pt"
URL_SERVER = "http://localhost:3000"
DRONE_IP = "12.12.12.13"
DRONE_SECRET = "DONT_SHARE_THIS_SECRET"
MISSION_NAME = "Apollo Mission 1"

# Set up a deque to store frames with a maximum length to simulate buffer overflow
frame_buffer = deque(maxlen=5)  # Adjust maxlen as needed to simulate overflow behavior

# Initialize model and retrieve detection rules
rules = get_drone_rules(URL_SERVER, DRONE_IP, DRONE_SECRET, MISSION_NAME)
print(f"Rules: {rules}")
model = YOLO(ML_MODEL).to("cuda" if DEVICE == "jetson" else "cpu")
class_names = model.names

# Capture and display thread function
def capture():
    cap = cv2.VideoCapture(0 if DEVICE != "jetson" else
                           "thetauvcsrc mode=4K ! queue! h264parse! nvv4l2decoder ! queue ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink"
                           )

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Add frame to the deque, automatically removing the oldest frame if maxlen is reached
        frame_buffer.append(frame)

        time.sleep(0.1)  # Introduce a delay to control overflow frequency

    cap.release()
    cv2.destroyAllWindows()

# Inference and sending detections thread function
def inference():
    while True:
        if len(frame_buffer) == 0:
            time.sleep(0.1)
            continue

        frame = frame_buffer.popleft()  # Get the oldest frame from the deque
        original_height, original_width = frame.shape[:2]
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

                # Calculate padding to make the fold dimensions divisible by 32
                pad_height = (32 - fold.shape[0] % 32) % 32
                pad_width = (32 - fold.shape[1] % 32) % 32
                pad_color = (0, 0, 0)  # Black padding
                fold_padded = cv2.copyMakeBorder(fold, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=pad_color)

                # Convert fold to tensor
                fold_tensor = torch.tensor(fold_padded, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                if DEVICE == "jetson":
                    fold_tensor = fold_tensor.to("cuda")

                # Run inference on the fold
                detections = model(fold_tensor)[0]

                # Scale factors for bounding box adjustments
                x_scale = (x_end - x_start) / original_width
                y_scale = (y_end - y_start) / original_height

                for data in detections.boxes.data.tolist():
                    confidence = data[4]
                    class_id = int(data[5])  # Class ID is stored in the 5th index
                    label = class_names[class_id]

                    # Check against rules
                    for rule_id, rule in rules.items():
                        if label == rule['object'] and confidence >= rule['confidence']['value']:
                            # Bounding box coordinates with scaling
                            xmin, ymin, xmax, ymax = (
                                int(data[0] * x_scale) + x_start,
                                int(data[1] * y_scale) + y_start,
                                int(data[2] * x_scale) + x_start,
                                int(data[3] * y_scale) + y_start,
                            )
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)

                            # Label and confidence text
                            text = f"{label}: {confidence:.2f}"
                            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

                            # Send detection data asynchronously
                            asyncio.run(send(frame, label, confidence))

                            break  # Exit once a matching rule is found

# Asynchronous function to send detection data
async def send(image, label, confidence):
    # Prepare encoded image for sending
    image_encoded = cv2.imencode('.jpg', image)[1].tobytes()

    # Create a FormData object to handle file upload
    form_data = aiohttp.FormData()
    form_data.add_field('image', image_encoded, filename='detected_image.jpg', content_type='image/jpeg')

    # Prepare the detection data as additional fields in the form
    description = get_description_from_image_with_groq_cloud(image_encoded)
    form_data.add_field('droneIp', DRONE_IP)
    form_data.add_field('secret', DRONE_SECRET)
    form_data.add_field('missionName', MISSION_NAME)
    form_data.add_field('detectedObject', label)
    form_data.add_field('confidence', str(confidence))  # Make sure it's a string
    form_data.add_field('description', description)

    # Send the POST request asynchronously
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{URL_SERVER}/api/detection/add", data=form_data) as response:
            if response.status == 200:
                response_data = await response.json()
                print("Response from server:", response_data)
            else:
                print(f"Failed to get a valid response. Status: {response.status}")

# Start threads
capture_thread = threading.Thread(target=capture)
inference_thread = threading.Thread(target=inference)

capture_thread.start()
inference_thread.start()

capture_thread.join()
inference_thread.join()
