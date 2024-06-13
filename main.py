from ultralytics import YOLO
import cv2
import torch
import cvzone
import math


device = torch.device("mps" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture('/Users/amananand/PycharmProjects/fire-detection/videos/fire.mp4')
#
# cap.set(3, 640)  # Set width
# cap.set(4, 640)  # Set height

model = YOLO('/Users/amananand/PycharmProjects/fire-detection/model/best (1).pt').to(device)

classnames =['fire','smoke']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 640))



    results = model(frame,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame, (x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])

            print(cls)

            cvzone.putTextRect(frame, f'{classnames[cls]} {conf}', (max(0, x1), max(35, y1)))

    cv2.imshow('Webcam Video', frame)
    cv2.waitKey(1)
