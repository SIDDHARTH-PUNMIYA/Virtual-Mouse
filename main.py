import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils
ctime = 0
ptime = 0
while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgrgb)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (0, 255, 255), cv2.FILLED) # this will mark the id =0 point

            mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (15, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("IMG ", img)
    cv2.waitKey(1)