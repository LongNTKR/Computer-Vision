import HandTracking as ht
import cv2

def main():
    detector = ht.HandDectector(path='D:\Code\Python\OpenCV\HandTracking\parameters.h5')
    
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        
        img = detector.find_hands_land_mark(img, draw=False)
        
        lmList = detector.find_position(img, draw=False)
        if len(lmList) != 0:
            img = detector.sign_digits_reconition(img)
        
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()

