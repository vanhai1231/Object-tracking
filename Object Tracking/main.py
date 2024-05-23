import cv2 as cv
from ultralytics import YOLO
from sort import Sort
import numpy as np

# Load YOLOv8 model
model = YOLO('C:\\Users\\PC\\Downloads\\Yolo_research-master\\Yolo_research-master\\yolov8n.pt')

# Open video
cap = cv.VideoCapture('C:\\Users\\PC\\Downloads\\highway.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

roi1_top_left = (600, 200)
roi1_bottom_right = (800, 400)

roi2_top_left = (800, 400)
roi2_bottom_right = (1000, 600)

# Initialize SORT tracker
tracker = Sort()

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame)

    # Get bounding boxes and object IDs
    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes)

    # Update tracker
    trackers = tracker.update(bboxes)

    # Draw bounding boxes on the original frame
    for d in trackers:
        x1, y1, x2, y2, obj_id = map(int, d)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, str(obj_id), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Draw the ROIs on the frame
    cv.rectangle(frame, roi1_top_left, roi1_bottom_right, (0, 0, 255), 2)
    cv.rectangle(frame, roi2_top_left, roi2_bottom_right, (0, 0, 255), 2)
    out.write(frame)

    # Display image
    cv.imshow('YOLO V8 Detection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
