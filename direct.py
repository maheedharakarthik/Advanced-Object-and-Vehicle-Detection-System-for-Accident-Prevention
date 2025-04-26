import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import RPi.GPIO as GPIO  # Import GPIO library

# Set number of threads via environment variable
os.environ["TFLITE_NUM_THREADS"] = "3"  # Adjust based on your Pi model (e.g., Raspberry Pi 4 has 4 cores)

# Set up GPIO pins for LEDs
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
LED_PINS = [8, 10, 11]  # GPIO Pins for LEDs (Pin 8, 10, 11)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)  # Set initial state to LOW (off)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="lite0-det-default.tflite")

# Allocate tensors
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Labels for COCO dataset
with open('labels.txt') as file:
    labels = {i: a.strip() for i, a in enumerate(file.readlines())}

# Set minimum confidence threshold for detections
min_conf_threshold = 0.35

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Initialize count dictionary
    count_by_class = {
        'person': 0,
        'car': 0,
        'motorcycle': 0,
        'bus': 0,
        'truck': 0
    }

    # Resize and preprocess the frame for the model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

    # Perform inference
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Process detections
    frame_height, frame_width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:
            class_id = int(classes[i])
            object_name = labels.get(class_id, str(class_id))

            ymin = int(max(1, boxes[i][0] * frame_height))
            xmin = int(max(1, boxes[i][1] * frame_width))
            ymax = int(min(frame_height, boxes[i][2] * frame_height))
            xmax = int(min(frame_width, boxes[i][3] * frame_width))

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Display label and confidence
            label = f'{object_name}: {int(scores[i] * 100)}%'
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0], label_ymin), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Count target classes
            if object_name in count_by_class:
                count_by_class[object_name] += 1

    # Measure inference time
    inference_time = time.time() - start_time
    cv2.putText(frame, f"Inference Time: {inference_time:.4f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show FPS
    fps = 1 / inference_time if inference_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display counts
    y_offset = 90
    for cls, cnt in count_by_class.items():
        cv2.putText(frame, f"{cls}: {cnt}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25

    # Update GPIO pins based on the detection counts
    # Control LEDs based on detected vehicles
    if count_by_class['car'] > 0:
        GPIO.output(8, GPIO.HIGH)  # Turn on LED for car
    else:
        GPIO.output(8, GPIO.LOW)  # Turn off LED for car

    if count_by_class['truck'] > 0:
        GPIO.output(10, GPIO.HIGH)  # Turn on LED for truck
    else:
        GPIO.output(10, GPIO.LOW)  # Turn off LED for truck

    if count_by_class['motorcycle'] > 0:
        GPIO.output(11, GPIO.HIGH)  # Turn on LED for motorcycle
    else:
        GPIO.output(11, GPIO.LOW)  # Turn off LED for motorcycle

    # Display the frame
    cv2.imshow('Vehicle Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up GPIO and webcam
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()  # Clean up GPIO resources
