import cv2
import cvzone
from ultralytics import YOLO
import math
from Project_2_sort_people_counting_helper import *
# Load the YOLO model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("videos/people.mp4")

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
mask = cv2.imread("images/project_2_mask.png")
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = set()
totalCountDown = set()

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
while True:
    b, frame = cap.read()
    if not b:
        break  # Break the loop if frame is not captured
    masked = cv2.bitwise_and(frame,mask)
    # Get results from the model
    results = model(masked, stream=True)
    imgGraphics = cv2.imread("images/project_2_graphics.png", cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, imgGraphics, (740, 250))
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
                if currentClass == "person" and conf > 0.3:
                    #cvzone.putTextRect(frame,f"{currentClass}",(max(0,x1),max(35,y1)), scale=0.6,offset = 3, thickness=1)
                    #cvzone.cornerRect(frame, bbox, l=9)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
    # Display the frame with detections
    resultstracker = tracker.update(detections)
    cv2.line(frame,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),color = (255,0,0),thickness=5)
    cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), color=(255, 0, 0), thickness=5)

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

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 25 < cy < limitsUp[3] + 25:
            if id not in totalCountUp:
                totalCountUp.add(id)
                cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        elif limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 25 < cy < limitsDown[3] + 25:
            if id not in totalCountDown:
                totalCountDown.add(id)
                cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
            # # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(frame, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(frame, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
    cv2.imshow("img",frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
