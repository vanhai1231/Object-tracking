import cv2 as cv
from ultralytics import YOLO
from sort import Sort
import numpy as np

# Load YOLOv8 model
model = YOLO('C:\\Users\\PC\\Downloads\\Yolo_research-master\\Yolo_research-master\\yolov8n.pt')

# Open webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: không thể mở webcam.")
    exit()

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Initialize SORT tracker
tracker = Sort()

roi1_top_left = (0, 0)
roi1_bottom_right = (frame_width // 2, frame_height)

roi2_top_left = (frame_width // 2, 0)
roi2_bottom_right = (frame_width, frame_height)

def calculate_intersection_area(x1, y1, x2, y2, roi_top_left, roi_bottom_right):
    roi_x1, roi_y1 = roi_top_left
    roi_x2, roi_y2 = roi_bottom_right

    intersect_x1 = max(x1, roi_x1)
    intersect_y1 = max(y1, roi_y1)
    intersect_x2 = min(x2, roi_x2)
    intersect_y2 = min(y2, roi_y2)

    if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
        return (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
    else:
        return 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: không thể đọc frame.")
        break

    results = model.track(frame)
    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes)

    # Update tracker
    trackers = tracker.update(bboxes)
    for d in trackers:
        x1, y1, x2, y2, obj_id = map(int, d)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, str(obj_id), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        center_x = (x1 + x2) // 2
        area_left = calculate_intersection_area(x1, y1, x2, y2, roi1_top_left, roi1_bottom_right)
        area_right = calculate_intersection_area(x1, y1, x2, y2, roi2_top_left, roi2_bottom_right)

        if area_left > area_right:
            print(f'Đối tượng ID {obj_id} đang ở vùng 1 (bên trái)')
        else:
            print(f'Đối tượng ID {obj_id} đang ở vùng 2 (bên phải)')

    cv.rectangle(frame, roi1_top_left, roi1_bottom_right, (0, 0, 255), 2)
    cv.rectangle(frame, roi2_top_left, roi2_bottom_right, (0, 0, 255), 2)

    out.write(frame)

    cv.imshow('YOLO V8 Detection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
