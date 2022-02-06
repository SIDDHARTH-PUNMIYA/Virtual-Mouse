import cv2
import mediapipe as mp
import time
import handtrackingmodule as htm
ctime = 0
ptime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findpos(img, draw=True)
    if len(lmlist) != 0:
        print(lmlist[4])
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (15, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("IMG ", img)
    cv2.waitKey(1)
