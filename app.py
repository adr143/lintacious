import time
import busio
import board
from datetime import datetime
import adafruit_amg88xx
import numpy as np
from scipy.ndimage import zoom
from ultralytics import YOLO  # YOLOv8
import cv2
import os
from flask import Flask, Response, render_template
from flask_sqlalchemy import SQLAlchemy
import gpiod
from gpiod.line import Direction, Value
import time

CHIP_PATH = "/dev/gpiochip0"  # Default GPIO chip
BUZZER_PIN = 18  # GPIO18

# Set up GPIO
line = gpiod.request_lines(
        CHIP_PATH,
        consumer="buzzer-control",
        config={
            BUZZER_PIN: gpiod.LineSettings(
                direction=Direction.OUTPUT, output_value=Value.INACTIVE
            )
        },
    )

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'thermal_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///records.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Initialize I2C and AMG8833
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_amg88xx.AMG88XX(i2c)

# Load YOLOv8 Model
model = YOLO("./Desktop/Thermal/lintacious/test1.pt")  # Replace with your trained model
#model = YOLO("./Desktop/Thermal/thermal_7.pt")

# Image processing settings
interp_factor = 8  # Upscale 8x8 to 64x64
scale_factor = 8  # Scale display for better visualization
confidence_threshold = 0.8  # Minimum confidence to detect objects
bbox_thickness = 1  # Thinner bounding box
text_offset = 10  # Offset for confidence display

# Class mapping
class_labels = {0: "Leech"}

class ThermalRecords(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    leech_count = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

def generate_frames():
    while True:
        # Temperature range for detection
        min_temp = 10
        max_temp = 35

        # Capture AMG8833 thermal data
        data = np.array(sensor.pixels)

        # Filter out values outside temperature range
        data = np.clip(data, min_temp, max_temp)

        # Upscale to 64x64
        data_interp = zoom(data, interp_factor)

        # Convert thermal data to grayscale (0-255)
        thermal_image = cv2.normalize(data_interp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a colormap for visualization
        thermal_image_colored = cv2.applyColorMap(thermal_image, cv2.COLORMAP_JET)

        # Convert to RGB (YOLOv8 expects 3-channel)
        thermal_image_rgb = cv2.cvtColor(thermal_image_colored, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference
        results = model(thermal_image_rgb)

        # Parse YOLO results
        boxes, scores, class_ids = [], [], []
        for r in results:
            for box in r.boxes:
                conf = box.conf[0].item()
                if conf < confidence_threshold:
                    continue

                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                class_ids.append(cls)

        # Draw bounding boxes
        leech_count = 0
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            if class_ids[i] == 0:
                leech_count += 1
                cv2.rectangle(thermal_image_colored, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)

        # Resize for better viewing
        large_thermal_image = cv2.resize(thermal_image_colored, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Create side panel
        side_panel = np.zeros((large_thermal_image.shape[0], 300, 3), dtype=np.uint8)
        cv2.putText(side_panel, f"Leeches: {leech_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display confidence percentages
        y_offset = 60
        for i, conf in enumerate(scores):
            if class_ids[i] == 0:
                conf_text = f"{conf * 100:.2f}%"
                cv2.putText(side_panel, conf_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30

        # Combine thermal image and side panel
        combined_image = np.hstack((large_thermal_image, side_panel))

        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", combined_image)
        if leech_count > 0:
            line.set_value(BUZZER_PIN, Value.ACTIVE)
            time.sleep(2)
            line.set_value(BUZZER_PIN, Value.INACTIVE)
            timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp_str}.jpg"
            filepath = os.path.join("Desktop", "Thermal", "static", "thermal_images", filename)
            print(filepath)
            cv2.imwrite(filepath, thermal_image_colored)
            with app.app_context():
                new_entry = ThermalRecords(filename=filepath, leech_count=int(leech_count))
                db.session.add(new_entry)
                db.session.commit()
            time.sleep(2)
        line.set_value(BUZZER_PIN, Value.INACTIVE)
        frame = buffer.tobytes()

        # Yield frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.5)  # Reduce CPU usage

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

# Flask route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/records')
def display_records():
    with app.app_context():
        images = ThermalRecords.query.all()
    return render_template('records.html', images=images)

# Run Flask app
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        line.release()
