import cv2
import cvzone
from ultralytics import YOLO
import math

# Load the YOLO model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("videos/ppe-1-1.mp4")

coco_dataset_classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    b, frame = cap.read()
    if not b:
        break  # Break the loop if frame is not captured

    # Get results from the model
    results = model(frame, stream=True)

    # Iterate through the results
    for r in results:
        # Ensure 'boxes' is accessed correctly
        if hasattr(r, 'boxes'):
            for box in r.boxes:
                #bounding box
                """print(box)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)"""

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1,y2-y1
                bbox = int(x1), int(y1), int(w), int(h)
                cvzone.cornerRect(frame,bbox)

                #confidence

                conf = box.conf[0]
                print(conf)

                #classname
                classed = coco_dataset_classNames[int(box.cls[0])]
                print(classed)
                cvzone.putTextRect(frame,f"{classed}",(max(0,x1),max(35,y1)), scale=1, thickness=1)
    # Display the frame with detections
    cv2.imshow('YOLO Object Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
