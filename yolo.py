import cv2
import numpy as np
import onnxruntime as ort
import time

THRESHOLD = .27

# Load YOLOv5n ONNX model
session = ort.InferenceSession("yolov5n.onnx")
input_name = session.get_inputs()[0].name

# Load COCO class labels
with open("labels.txt") as file:
    labels = [line.strip() for line in file.readlines()]
    target_classes = {i: labels[i] for i in [0, 1, 2, 3, 5, 7]}

# Start webcam
cap = cv2.VideoCapture(0)

prev_time = time.time()

# Preprocessing
def preprocess(image):
    resized = cv2.resize(image, (640, 640))
    img = resized[..., ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0
    return img

# Postprocessing with NMS
def postprocess(outputs, img_shape):
    preds = outputs[0][0]  # (25200, 85)

    boxes, scores, class_ids = [], [], []
    for pred in preds:
        conf = float(pred[4])
        if conf > THRESHOLD:
            class_scores = pred[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            score = conf * class_conf
            if score > THRESHOLD and class_id in target_classes:
                cx, cy, w, h = pred[0:4]
                x1 = int((cx - w / 2) * img_shape[1] / 640)
                y1 = int((cy - h / 2) * img_shape[0] / 640)
                x2 = int((cx + w / 2) * img_shape[1] / 640)
                y2 = int((cy + h / 2) * img_shape[0] / 640)
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, THRESHOLD, 0.45)
    final_boxes, final_scores, final_class_ids = [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])

    return final_boxes, final_scores, final_class_ids

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    boxes, scores, class_ids = postprocess(outputs, frame.shape)

    count_by_class = {name: 0 for name in target_classes.values()}

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{target_classes[class_id]}: {int(score * 100)}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        count_by_class[target_classes[class_id]] += 1

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display counts
    y = 30
    for label, count in count_by_class.items():
        cv2.putText(frame, f"{label}s: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += 25

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Draw top-right legend circles using groupings
    top_right_colors = {
        (0, 0, 255): ['car'],              # Red
        (0, 255, 0): ['truck'],           # Green
        (255, 0, 0): ['person', 'motorcycle']  # Blue
    }

    radius = 12
    spacing = 40
    base_x = frame.shape[1] - 30
    base_y = 30

    i = 0
    for color, labels_group in top_right_colors.items():
        total_count = sum(count_by_class.get(label, 0) for label in labels_group)
        center = (base_x, base_y + i * spacing)
        thickness = -1 if total_count > 0 else 2
        cv2.circle(frame, center, radius, color, thickness)
        label_text = "/".join(label.capitalize() for label in labels_group)
        cv2.putText(frame, label_text, (center[0] - 120, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        i += 1

    cv2.imshow("YOLOv5n Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()