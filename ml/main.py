from ultralytics import YOLO
import torch
import cv2

# Define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
DEVICE = "pc"  # Use 'jetson' for the RICOH THETA X with Jetson Nano, or 'pc' for a regular webcam
NUM_FOLDS = 2  # Change this value to increase the number of folds for division (e.g., 2, 4, 8)

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
    model = YOLO("yolov8n.pt").to("cuda")  # Ensure the model runs on CUDA
else:
    model = YOLO("yolov8n.pt")

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

            # Resize the fold to 640x640 for YOLO inference
            fold_resized = cv2.resize(fold, (640, 640), interpolation=cv2.INTER_AREA)

            # Convert the fold to a tensor
            fold_tensor = torch.tensor(fold_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            # Move the fold tensor to GPU if using Jetson
            if DEVICE == "jetson":
                fold_tensor = fold_tensor.to("cuda")

            # Run the YOLO model on the fold (perform inference)
            detections = model(fold_tensor)[0]

            # Scale factors for bounding box adjustments
            x_scale = (x_end - x_start) / 640
            y_scale = (y_end - y_start) / 640

            for data in detections.boxes.data.tolist():
                # Extract the confidence associated with the detection
                confidence = data[4]

                # Filter out weak detections by ensuring the confidence is greater than the minimum
                if float(confidence) < CONFIDENCE_THRESHOLD:
                    continue

                # Get the class ID and label
                class_id = int(data[5])  # The class ID is stored in the 5th index
                label = class_names[class_id]  # Get the label from class names

                # Draw the bounding box on the fold if confidence is sufficient
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

    # Display the frame with detections
    cv2.imshow("Input", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
