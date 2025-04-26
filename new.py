import cv2
import numpy as np
import tensorflow as tf
import time
import argparse
import os

def load_labels(labels_path):
    labels = {}
    try:
        with open(labels_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                labels[i+1] = line.strip()
        print(f"Loaded {len(labels)} labels from {labels_path}")
    except FileNotFoundError:
        labels = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        print("Labels file not found. Using default vehicle labels.")
    return labels

def run_detection(saved_model_dir, labels_path=None, confidence_threshold=0.5, camera_id=0):
    print(f"Loading SavedModel from: {saved_model_dir}")

    try:
        model = tf.saved_model.load(saved_model_dir)
        infer = model.signatures['serving_default']
        print("Model loaded successfully")

        input_tensor_spec = infer.structured_input_signature[1]
        input_tensor_name = list(input_tensor_spec.keys())[0]
        input_shape = input_tensor_spec[input_tensor_name].shape

        if input_shape[1] is None or input_shape[2] is None:
            height, width = 320, 320
        else:
            height, width = input_shape[1], input_shape[2]

        print(f"Using image size: {height}x{width}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    labels = load_labels(labels_path) if labels_path else {
        2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
    }

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (ID: {camera_id}).")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    print("Press 'q' to quit")

    start_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))

        input_tensor = tf.convert_to_tensor(frame_resized, dtype=tf.uint8)
        input_tensor = tf.expand_dims(input_tensor, 0)

        try:
            start_infer = time.time()
            detections = infer(input_tensor)
            infer_time = time.time() - start_infer

            boxes = detections['detection_boxes'].numpy()[0]
            scores = detections['detection_scores'].numpy()[0]
            classes = detections['detection_classes'].numpy()[0].astype(np.int32)

            for i in range(len(scores)):
                if scores[i] >= confidence_threshold:
                    class_id = int(classes[i])
                    if class_id in labels:
                        ymin, xmin, ymax, xmax = boxes[i]
                        ymin, xmin, ymax, xmax = (ymin * frame_height, xmin * frame_width,
                                                  ymax * frame_height, xmax * frame_width)

                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                        object_name = labels.get(class_id, f"Class {class_id}")
                        label = f"{object_name}: {int(scores[i] * 100)}%"

                        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (xmin, ymin - label_size[1] - 10),
                                      (xmin + label_size[0], ymin), (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, label, (xmin, ymin - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            print(f"Error during inference: {e}")
            cv2.putText(frame, "Inference error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {infer_time * 1000:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Vehicle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run object detection using TensorFlow SavedModel')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the SavedModel directory')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels file (one label per line)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold (0.0 to 1.0)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        exit(1)

    if not os.path.exists(os.path.join(args.model_dir, "saved_model.pb")):
        print(f"Error: No saved_model.pb found in {args.model_dir}")
        print("This script requires a TensorFlow SavedModel format.")
        exit(1)

    run_detection(args.model_dir, args.labels, args.threshold, args.camera)
