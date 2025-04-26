import cv2
import numpy as np
import tensorflow as tf
import time

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="lite0-det-default.tflite")
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
    # Capture frame from webcamw
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

    # Draw legend circles with conditional fill
    colors = {
        'car': (0, 0, 255),
        'truck': (0, 255, 0),
        'motorcycle': (255, 0, 0)
    }
    y_positions = {
        'car': 30,
        'truck': 60,
        'motorcycle': 90
    }
    for vehicle, color in colors.items():
        pos_y = y_positions[vehicle]
        thickness = -1 if count_by_class[vehicle] > 0 else 2
        cv2.circle(frame, (frame_width - 130, pos_y), 10, color, thickness)
        cv2.putText(frame, vehicle.capitalize(), (frame_width - 115, pos_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Show FPS
    fps = 1 / (time.time() - start_time) if 'start_time' in locals() else 0
    start_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display counts
    y_offset = 60
    for cls, cnt in count_by_class.items():
        cv2.putText(frame, f"{cls}: {cnt}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25

    # Display the frame
    cv2.imshow('Vehicle Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()