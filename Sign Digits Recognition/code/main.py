import cv2 as cv
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyautogui
from scipy import stats as st
import os
import time
class HandDectector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, dectectionCon=0.5, trackCon=0.5,
                 path='E:\\FPT\\2023 SP\\DAP\\parameters.h5'):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.dectectionCon = dectectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, 
                                        self.dectectionCon, self.trackCon)
        
        self.model = tf.keras.models.load_model(path, compile=False)
        self.class_names = [0, 1, 2, 3, 4, 5]
        self.lmList = []
        
    def find_hands_land_mark(self, img, draw=False, connect=False):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks is not None and draw:
            handLms = self.results.multi_hand_landmarks[-1]
            if connect:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            else:
                self.mpDraw.draw_landmarks(img, handLms)
    
    def find_position(self, img, draw=False, size_cricle=8):
        lmList = [] 
        if self.results.multi_hand_landmarks is not None:
            handLms = self.results.multi_hand_landmarks[-1]
            res = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                res.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), size_cricle, (0, 0, 255), -1)
            lmList.append(np.array(res))
        self.lmList = lmList
        return lmList
    
    def _relu(self, x):
        return x if x > 0 else 0  
                
    def sign_digits_reconition(self, img, bias=25):
        handLms = self.lmList[-1]
        x_min = self._relu(np.min(handLms[:,1]) - bias)
        x_max = self._relu(np.max(handLms[:,1]) + bias)
        y_min = self._relu(np.min(handLms[:,2]) - bias)
        y_max = self._relu(np.max(handLms[:,2]) + bias)
        masks = cv.rectangle(np.zeros(img.shape[:2], dtype='uint8'), (x_min, y_min), (x_max, y_max), 255, -1)
        hand = cv.resize(cv.bitwise_and(img, img, mask=masks)[y_min:y_max, x_min:x_max], 
                        (128, 128), interpolation = cv.INTER_LINEAR)
        input = np.array(hand).reshape(1, 128, 128, 3)
        feature = self.model(input)
        predict = tf.math.argmax(tf.nn.softmax(feature, axis=1), axis=1)[0]
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        return predict

def findButton():
    time.sleep(3)
    path = r'E:\FPT\2023 SP\DAP'
    skip_forward = pyautogui.locateCenterOnScreen(os.path.join(path,'increase.PNG'))
    skip_back = pyautogui.locateCenterOnScreen(os.path.join(path,'decrease.PNG'))
    window = pyautogui.locateCenterOnScreen(os.path.join(path,'media_player.PNG'))
    play_plause = pyautogui.locateCenterOnScreen(os.path.join(path,'start_stop.PNG'))
    return skip_forward, skip_back, window, play_plause

cap = cv.VideoCapture(0)
dectector = HandDectector()
cv.namedWindow('Camera')
con = False
predicts = []
skip_forward, skip_back, window, play_plause = findButton()
while True:
    success, frame = cap.read()
    if not success:
        continue
    dectector.find_hands_land_mark(frame)
    lmList = dectector.find_position(frame)
    if lmList:
        predict = dectector.sign_digits_reconition(frame)
        if len(predicts) < 10:
            predicts.append(predict)
        else:
            mode = st.mode(predicts)[0]
            predicts = []
            if mode == 0:
                con = True
            else:
                if mode == 1 and con:
                    pyautogui.moveTo(skip_forward)
                    pyautogui.click()
                elif mode == 2 and con:
                    pyautogui.moveTo(skip_back)
                    pyautogui.click()
                elif mode == 3 and con:
                    pyautogui.press(['volumeup']*5)
                elif mode == 4 and con:
                    pyautogui.press(['volumedown']*5)
                elif mode == 5 and con:
                    pyautogui.moveTo(play_plause)
                    pyautogui.click()
                con = False
                
    cv.imshow('Camera', cv.flip(frame, 1))
    if cv.waitKey(5) & 0xFF == 27:
        break