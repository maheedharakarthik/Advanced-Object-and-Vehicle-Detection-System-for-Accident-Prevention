import cv2
import numpy as np
import tensorflow as tf
import time
import requests

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

# Set minimum confidence threshold
min_conf_threshold = 0.35

# Raspberry Pi endpoints
raspberry_pi_ip = "http://192.168.1.100:5000"  # Change to your Pi IP
image_url = f"{raspberry_pi_ip}/image"
led_url = f"{raspberry_pi_ip}/change_status"

print("Press 'q' to quit")

while True:
    try:
        response = requests.get(image_url, timeout=2)
        if response.status_code != 200:
            print("Failed to get image from Raspberry Pi.")
            continue
        image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error:", e)
        continue

    if frame is None:
        print("Error: Received empty frame.")
        continue

    count_by_class = {
        'person': 0,
        'car': 0,
        'motorcycle': 0,
        'bus': 0,
        'truck': 0
    }

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    frame_height, frame_width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:
            class_id = int(classes[i])
            object_name = labels.get(class_id, str(class_id))

            ymin = int(max(1, boxes[i][0] * frame_height))
            xmin = int(max(1, boxes[i][1] * frame_width))
            ymax = int(min(frame_height, boxes[i][2] * frame_height))
            xmax = int(min(frame_width, boxes[i][3] * frame_width))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            label = f'{object_name}: {int(scores[i] * 100)}%'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0], label_ymin), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if object_name in count_by_class:
                count_by_class[object_name] += 1

    # Send LED update based on object counts
    try:
        led_status = {
            "8": count_by_class['car'] > 0,                           # LED 1 ON if car detected
            "10": count_by_class['truck'] > 0,                        # LED 2 ON if truck detected
            "11": (count_by_class['person'] + count_by_class['motorcycle']) > 0  # LED 3 ON if person or motorcycle
        }
        requests.post(led_url, json=led_status, timeout=1)
    except Exception as e:
        print("Failed to update LED status:", e)

    # Draw legend
    colors = {'car': (0, 0, 255), 'truck': (0, 255, 0), 'motorcycle': (255, 0, 0)}
    y_positions = {'car': 30, 'truck': 60, 'motorcycle': 90}
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

    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
