import numpy as np
import cv2
from ultralytics import YOLOv10
import supervision as sv

# Define object specific variables
dist = 0
focal = 450
width = 4

# Function to get distance from the camera
def get_dist(rect_params, image):
    # Extract number of pixels covered by rectangle
    pixels = rect_params[1][0]
    
    # Calculate distance
    dist = (width * focal) / pixels
    
    # Write distance on the image
    image = cv2.putText(image, f'Drone Z: {dist:.2f} cm', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (0, 0, 255), 2, cv2.LINE_AA)

    return dist

# Load YOLOv10 model
model_path = "C:/Users/JAK/Downloads/drone.pt"  # Update with your actual path
model = YOLOv10(model_path)

# Create bounding box and label annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialize video capture for drone camera (assuming it's camera index 1)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Unable to access webcam")
    exit()

# Basic constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')

# Create named window
cv2.namedWindow('Drone Detection & Distance Measurement', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Drone Detection & Distance Measurement', 800, 600)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Predefined mask for green color detection (adjust as needed)
    lower = np.array([37, 51, 24])
    upper = np.array([83, 104, 131])
    mask = cv2.inRange(hsv_img, lower, upper)

    # Remove noise from image
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Find contours in the image
    contours, _ = cv2.findContours(d_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    for cnt in contours:
        # Check contour area
        if 100 < cv2.contourArea(cnt) < 306000:
            # Draw a rectangle around the contour
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (255, 0, 0), 3)
            
            # Get distance from the camera
            dist = get_dist(rect, frame)

            # Calculate x and y coordinates of the drone with respect to the camera
            # Assuming the center of the frame as the reference point
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            
            drone_x = (rect[0][0] - center_x) * (dist / focal)
            drone_y = (rect[0][1] - center_y) * (dist / focal)
            
            # Display coordinates
            cv2.putText(frame, f'Drone X: {drone_x:.2f} cm', (10, 100), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Drone Y: {drone_y:.2f} cm', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    # Perform inference with YOLOv10
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate image with bounding boxes and labels
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Display annotated image with distance and coordinates
    cv2.imshow('Drone Detection & Distance Measurement', annotated_image)

    # Exit on 'Esc' key press
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
