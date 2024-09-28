import cv2
import cvzone
from ultralytics import YOLO
import math
from Project_1_sort_car_counting_helper import *
# Load the YOLO model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("videos/cars.mp4")

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
mask = cv2.imread("images/project_1_mask.png")
limits = [400, 297, 673, 297]
totalCount = set()

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
while True:
    b, frame = cap.read()
    if not b:
        break  # Break the loop if frame is not captured
    masked = cv2.bitwise_and(frame,mask)
    # Get results from the model
    results = model(masked, stream=True)
    imgGraphics = cv2.imread("images/project_1_graphics.png", cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, imgGraphics, (0, 0))
    detections = np.empty((0, 5))

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


                #confidence
                conf = box.conf[0]
                print(conf)
                #classname
                currentClass = coco_dataset_classNames[int(box.cls[0])]
                print(currentClass)
                if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                        or currentClass == "motorbike" and conf > 0.3:
                    #cvzone.putTextRect(frame,f"{currentClass}",(max(0,x1),max(35,y1)), scale=0.6,offset = 3, thickness=1)
                    #cvzone.cornerRect(frame, bbox, l=9)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
    # Display the frame with detections
    resultstracker = tracker.update(detections)
    cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),color = (255,0,0),thickness=5)
    for result in resultstracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx,cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if id not in totalCount:
                totalCount.add(id)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(frame,f" : {len(totalCount)}",(200,100),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=4,thickness=2,color=(50,50,255))

    cv2.imshow("img",frame)
    cv2.waitKey(0)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
