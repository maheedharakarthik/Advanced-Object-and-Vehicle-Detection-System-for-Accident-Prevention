import cv2
import numpy as np

# Load YOLOv5 ONNX model using OpenCV
net = cv2.dnn.readNet("yolov5n.onnx")

# Load COCO class labels
with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Preprocessing
def preprocess(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (640, 640), (0, 0, 0), True, crop=False)
    return blob

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    blob = preprocess(frame)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getLayers()]

    # Run inference
    detections = net.forward(output_layers)

    # Process the results
    # Similar postprocessing as in the previous method

    # Show the frame
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
