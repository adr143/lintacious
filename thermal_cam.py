import time
import busio
import board
import adafruit_amg88xx
import numpy as np
import cv2
from scipy.ndimage import zoom
from ultralytics import YOLO  # YOLOv8

# Initialize I2C and AMG8833
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_amg88xx.AMG88XX(i2c)

# Load YOLOv8 Model
model = YOLO("thermal_1.pt")  # Replace with your trained model

# Interpolation factor (for upscaling 8x8 to higher resolution)
interp_factor = 8  # Upscale to 64x64
frame_size = 64  # YOLO requires larger images, using 64x64 size

# Define the scaling factor for enlarging the live feed window
scale_factor = 8  # Scale the feed to 2x the original size

# Bounding box thickness
bbox_thickness = 1  # Thinner bounding box

# Text offset for class name
text_offset = 10  # Distance to the right of the bounding box

# Threshold for detection confidence
confidence_threshold = 0.95  # Only consider detections with confidence > 0.8

# Mapping class 0 to "Leech"
class_labels = {
    0: "Leech",  # Renaming class 0 to "Leech"
}

# Non-Maximum Suppression (NMS) function
def non_max_suppression(boxes, scores, iou_threshold=0.4):
    indices = np.argsort(scores)[::-1]  # Sort boxes by score
    final_boxes = []
    
    while len(indices) > 0:
        current_index = indices[0]
        current_box = boxes[current_index]
        final_boxes.append(current_box)
        
        remaining_indices = indices[1:]
        remaining_boxes = boxes[remaining_indices]
        
        iou_values = compute_iou(current_box, remaining_boxes)
        
        indices = indices[1:][iou_values < iou_threshold]
    
    return final_boxes

def compute_iou(box1, boxes):
    x1, y1, x2, y2 = box1
    area1 = (x2 - x1) * (y2 - y1)
    
    x1_boxes, y1_boxes, x2_boxes, y2_boxes = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area_boxes = (x2_boxes - x1_boxes) * (y2_boxes - y1_boxes)
    
    inter_x1 = np.maximum(x1, x1_boxes)
    inter_y1 = np.maximum(y1, y1_boxes)
    inter_x2 = np.minimum(x2, x2_boxes)
    inter_y2 = np.minimum(y2, y2_boxes)
    
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    
    union_area = area1 + area_boxes - inter_area
    iou = inter_area / union_area
    
    return iou

while True:
    # Capture AMG8833 thermal data (8x8 matrix)
    data = np.array(sensor.pixels)

    # Upscale to 64x64 using interpolation
    data_interp = zoom(data, interp_factor)

    # Convert thermal data to grayscale (0-255)
    thermal_image = cv2.normalize(data_interp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap to the thermal image (e.g., JET) to make it colorful
    thermal_image_colored = cv2.applyColorMap(thermal_image, cv2.COLORMAP_JET)

    # Convert grayscale to 3-channel image (YOLOv8 expects 3-channel)
    thermal_image_rgb = cv2.cvtColor(thermal_image_colored, cv2.COLOR_BGR2RGB)

    # Run YOLOv8 inference
    results = model(thermal_image_rgb)  # Results object holds all detections

    # Create a list to store boxes, scores, and class ids
    boxes = []
    scores = []
    class_ids = []

    for r in results:  # Loop through YOLO results (there could be multiple outputs)
        for box in r.boxes:  # Loop through each detection box in the output
            conf = box.conf[0].item()  # Confidence score
            if conf < confidence_threshold:  # Skip detections below threshold
                continue

            cls = int(box.cls[0].item())  # Class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Collect the bounding box and confidence score
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(cls)
    
    # Convert to numpy arrays for easy manipulation
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Apply Non-Maximum Suppression (NMS)
    final_boxes = non_max_suppression(boxes, scores, iou_threshold=0.4)

    # Initialize detection counts
    leech_count = 0

    # Draw the remaining bounding boxes after NMS and count "Leeches"
    for box in final_boxes:
        x1, y1, x2, y2 = map(int, box)

        # Count the number of leeches detected
        leech_count += 1  # Each box left after NMS is considered a leech

        # Draw the bounding box and label
        cv2.rectangle(thermal_image_colored, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)
        cv2.putText(thermal_image_colored, class_labels.get(0, "Leech"), 
                    (x1, y1 - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Resize the image to make it larger for display
    large_thermal_image = cv2.resize(thermal_image_colored, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Create a new image to display the class counts on the side
    side_panel = np.zeros((large_thermal_image.shape[0], 300, 3), dtype=np.uint8)  # Space for counts

    # Display the count of "Leeches" on the side panel
    y_offset = 20  # Starting position for class names
    text = f"Leeches: {leech_count}"  # Show the number of "Leeches"
    cv2.putText(side_panel, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Combine the side panel with the thermal image
    combined_image = np.hstack((large_thermal_image, side_panel))

    # Show YOLOv8 detection output with the class count panel
    cv2.imshow("YOLOv8 Thermal Detection", combined_image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
