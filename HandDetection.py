import cv2
import mediapipe as mp
import time
import autopy
import numpy as np
import pyautogui as pa
import math


class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, lms, self.mphands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):

        lml = []

        if self.results.multi_hand_landmarks:
            for lms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(lms.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lml.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return lml

    def FingersUp(self, landMarks):
        pos = []
        TipID = [4, 8, 12, 16, 20]

        if landMarks[TipID[0]][1] > landMarks[TipID[0] - 1][1]:
            pos.append(1)
        else:
            pos.append(0)

        for i in range(1, 5):
            if landMarks[TipID[i]][2] < landMarks[TipID[i] - 3][2]:
                pos.append(1)
            else:
                pos.append(0)

        print(pos)
        return pos

    def dist(self, x1, y1, x2, y2):
        d = abs((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        d = int(math.sqrt(d))
        print(d)
        return (d)


def main():
    px, py = 0, 0
    cx, cy = 0, 0
    w, h = autopy.screen.size()
    print("W= {} , H= {}".format(w, h))
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    d = handDetector()
    while True:
        Success, img = cap.read()
        img = d.findHands(img)
        lm = d.findPosition(img)
        drag = -1
        if len(lm) > 0:
            drag = d.dist(lm[4][1], lm[4][2], lm[8][1], lm[8][2])
            pos = d.FingersUp(lm)
            dm = -1
            x1, y1 = lm[8][1:]
            x2, y2 = lm[12][1:]
            x3 = np.interp(x1, (75, 640 - 75), (0, w + 400))
            y3 = np.interp(y1, (75, 480 - 75), (0, h + 100))
            # print(pos)

            if drag < 21:
                dm = 0
                ch = 1
                if dm == 0 and ch == 1:
                    pa.mouseDown()
                cx = px + (x3 - px) / 7
                cy = py + (y3 - py) / 7
                if w - cx < 1280 and w - cx > 0 and cy < 720:
                    autopy.mouse.move(abs(w - cx), cy)
                px, py = cx, cy
                ch += 1

            if dm == -1:
                ch = 0
                if dm == -1 and ch == 0:
                    pa.mouseUp()

            if pos[1] == 1 and pos[2] == 0 and pos[0] == 1:
                cx = px + (x3 - px) / 7
                cy = py + (y3 - py) / 7
                if w - cx < 1280 and w - cx > 0 and cy < 720:
                    autopy.mouse.move(abs(w - cx), cy)
                px, py = cx, cy
                # print('Mouse Mode')
            elif pos[0] == 0 and pos[2] == 0 and pos[1] != 0 and w - cx < 1280:
                autopy.mouse.click()
                # print('Click')

            elif pos[0] == 1 and pos[1] == 1 and pos[2] == 1 and w - cx < 1280:
                pa.scroll(20)

            elif pos[0] == 1 and pos[1] == 0 and pos[2] == 0 and w - cx < 1280:
                pa.scroll(-20)

            elif pos[0] == 0 and pos[1] == 0 and pos[2] == 0 and pos[3] == 0 and pos[4] == 0:
                break

        img = cv2.flip(img, 1)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        # cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
