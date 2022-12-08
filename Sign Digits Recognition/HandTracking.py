import mediapipe as mp
import tensorflow as tf
import cv2
import time
import numpy as np

class HandDectector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, dectectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.dectectionCon = dectectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, 
                                        self.dectectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.model = tf.keras.models.load_model('D:\Code\Python\OpenCV\HandTracking\parameters.h5')
        self.class_names = [0, 1, 2, 3, 4, 5]
        
    def find_hands_land_mark(self, img, draw=True, connect=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    if connect:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    else:
                        self.mpDraw.draw_landmarks(img, handLms)
        return img

    def find_position(self, img, draw=True, size_cricle=8):
        lmList = [] 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                res = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    res.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), size_cricle, (0, 0, 255), -1)
                lmList.append(np.array(res))
        self.lmList = lmList
        return self.lmList
    
    def distance(self, x, y):
        return int(((x[1] - y[1])**2 + (x[2] - y[2])**2)**0.5)
    
    def _relu(self, x):
        return x if x > 0 else 0
    
    def sign_digits_reconition(self, img, bias=25, pos=(20, 80), ins = 0):
        for handLms in self.lmList:
            x_min = self._relu(np.min(handLms[:,1])-bias)
            x_max = self._relu(np.max(handLms[:,1])+bias)
            y_min = self._relu(np.min(handLms[:,2])-bias)
            y_max = self._relu(np.max(handLms[:,2])+bias)
            start = (x_min, y_min)
            end = (x_max, y_max)
            masks = cv2.rectangle(np.zeros(img.shape[:2], dtype='uint8'), start, end, 255, -1)
            hand = np.array(cv2.resize(cv2.bitwise_and(img, img, mask=masks)[y_min:y_max, x_min:x_max], 
                            (128, 128), interpolation = cv2.INTER_LINEAR)).reshape(1, 128, 128, 3)
            feature = self.model(hand)
            predict = tf.math.argmax(tf.nn.softmax(feature, axis=1), axis=1)
            cv2.putText(img, str(self.class_names[int(predict)]), (pos[0] + ins, pos[1]), 
                        cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
            cv2.rectangle(img, start, end, (255, 0, 0), 2)
            ins += 50
        return img
    
def main(fps=False):
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    
    dectector = HandDectector()

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        
        img = dectector.find_hands_land_mark(img, draw=False)
        
        lmList = dectector.find_position(img, draw=False)
        
        if len(lmList) != 0:
            img = dectector.sign_digits_reconition(img)
            
        if fps:    
            cTime = time.time()
            fps = 1 / (cTime-pTime)
            pTime = cTime 
            cv2.putText(img, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
        
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()