import cv2
import handsDet as detector
# import mediapipe as detector
import time


# out = cv2.VideoWriter('ringHand.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30,
#  (640,480))

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.xOld, self.yOld = 0, 0
        self.xmOld, self.ymOld = 0, 0

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = detector.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, max_num_hands = self.maxHands,
                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        return img

    def findPosition(self, img, handNo=-1):

        lmList = []
        if self.results.multi_hand_landmarks:
            for nm in range(len((self.results.multi_hand_landmarks))):
                myHand = self.results.multi_hand_landmarks[nm]
                # print(self.results.multi_hand_world_landmarks[nm])
                mrk = list(enumerate(myHand.landmark))


                id, lm = mrk[14]
                id, lm2 = mrk[13]
                h, w, c = img.shape

                # print (mrk[0][1])
                cv2.line(img, (int(mrk[0][1].x * w), int(mrk[0][1].y * h)), (int(mrk[1][1].x * w), int(mrk[1][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[1][1].x * w), int(mrk[1][1].y * h)), (int(mrk[2][1].x * w), int(mrk[2][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[2][1].x * w), int(mrk[2][1].y * h)), (int(mrk[3][1].x * w), int(mrk[3][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[3][1].x * w), int(mrk[3][1].y * h)), (int(mrk[4][1].x * w), int(mrk[4][1].y * h)), (255, 255, 0), 8)

                cv2.line(img, (int(mrk[0][1].x * w), int(mrk[0][1].y * h)), (int(mrk[5][1].x * w), int(mrk[5][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[5][1].x * w), int(mrk[5][1].y * h)), (int(mrk[6][1].x * w), int(mrk[6][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[6][1].x * w), int(mrk[6][1].y * h)), (int(mrk[7][1].x * w), int(mrk[7][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[7][1].x * w), int(mrk[7][1].y * h)), (int(mrk[8][1].x * w), int(mrk[8][1].y * h)), (255, 255, 0), 8)

                cv2.line(img, (int(mrk[9][1].x * w), int(mrk[9][1].y * h)), (int(mrk[10][1].x * w), int(mrk[10][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[10][1].x * w), int(mrk[10][1].y * h)), (int(mrk[11][1].x * w), int(mrk[11][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[11][1].x * w), int(mrk[11][1].y * h)), (int(mrk[12][1].x * w), int(mrk[12][1].y * h)), (255, 255, 0), 8)

                cv2.line(img, (int(mrk[13][1].x * w), int(mrk[13][1].y * h)), (int(mrk[14][1].x * w), int(mrk[14][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[14][1].x * w), int(mrk[14][1].y * h)), (int(mrk[15][1].x * w), int(mrk[15][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[15][1].x * w), int(mrk[15][1].y * h)), (int(mrk[16][1].x * w), int(mrk[16][1].y * h)), (255, 255, 0), 8)

                cv2.line(img, (int(mrk[0][1].x * w), int(mrk[0][1].y * h)), (int(mrk[17][1].x * w), int(mrk[17][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[17][1].x * w), int(mrk[17][1].y * h)), (int(mrk[18][1].x * w), int(mrk[18][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[18][1].x * w), int(mrk[18][1].y * h)), (int(mrk[19][1].x * w), int(mrk[19][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[19][1].x * w), int(mrk[19][1].y * h)), (int(mrk[20][1].x * w), int(mrk[20][1].y * h)), (255, 255, 0), 8)

                cv2.line(img, (int(mrk[5][1].x * w), int(mrk[5][1].y * h)), (int(mrk[9][1].x * w), int(mrk[9][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[9][1].x * w), int(mrk[9][1].y * h)), (int(mrk[13][1].x * w), int(mrk[13][1].y * h)), (255, 255, 0), 8)
                cv2.line(img, (int(mrk[13][1].x * w), int(mrk[13][1].y * h)), (int(mrk[17][1].x * w), int(mrk[17][1].y * h)), (255, 255, 0), 8)


                xRec = int(min(mrk[5][1].x  * w, mrk[17][1].x  * w, mrk[0][1].x  * w) * 0.9)
                xRecM = int(max(mrk[5][1].x  * w, mrk[17][1].x  * w, mrk[0][1].x  * w) * 1.1)
                yRec = int(min(mrk[5][1].y * h, mrk[17][1].y * h, mrk[0][1].y * h) * 0.9)
                yRecM = int(max(mrk[5][1].y * h, mrk[17][1].y * h, mrk[0][1].y * h) * 1.1)


                cv2.rectangle(img, (xRec, yRec), (xRecM, yRec + xRecM - xRec), (255,0,0), 2)


                cv2.putText(img, self.results.multi_handedness[nm].classification[0].label, (xRec, yRec),
                 cv2.FONT_HERSHEY_PLAIN, 2,
                 (256, 128, 128), 2)

                

                for point in mrk:
                    id, lm = point
                    # print(lm.x // 1, lm.y// 1, lm.z)
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), int(6), (255, 255, 255), cv2.FILLED)


        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(mode=False)
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        # cv2.imshow("cap", img)
        # cv2.waitKey(0)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        # if len(lmList) != 0:
        #     print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        # out.write(img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()