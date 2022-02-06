import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, node=False, maxhands=2, detectconfd=0.5, trackconfd=0.5):
        self.node = node
        self.maxhands = maxhands
        self.detectconfd = detectconfd
        self.trackconfd = trackconfd
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.node, self.maxhands, self.trackconfd, self.trackconfd)
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)
        return img

    def findpos(self, img, handno=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]

            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)  # this will mark the id =0 point
        return lmList


def main():
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findpos(img, draw=True)
        if len(lmlist)!=0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (15, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("IMG ", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
