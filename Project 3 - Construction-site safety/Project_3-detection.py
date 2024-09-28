import cv2
import cvzone
from ultralytics import YOLO
import math
from Project_1_sort_car_counting_helper import *
# Load the YOLO model
model = YOLO('best.pt')
cap = cv2.VideoCapture("../videos/ppe-1-1.mp4")


names =  ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']


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


                #confidence
                conf = box.conf[0]
                print(conf)
                #classname
                currentClass = names[int(box.cls[0])]
                print(currentClass)
                if conf > 0.3:
                    cvzone.putTextRect(frame,f"{currentClass}",(max(0,x1),max(35,y1)), scale=1,offset = 3, thickness=1)
                    cvzone.cornerRect(frame, bbox, l=9)

    cv2.imshow("img",frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
