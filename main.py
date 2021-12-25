import cv2 as cv
import numpy as np
import time

face_path = 'cascades/haarcascade_frontalface_default.xml'
face_detector = cv.CascadeClassifier(face_path)
eye_path = 'cascades/haarcascade_eye.xml'
eye_detector = cv.CascadeClassifier(eye_path)

def detectFace():
    rects = face_detector.detectMultiScale(gray, 
        scaleFactor=1.1,
        minNeighbors=5, 
        minSize=(30, 30), 
        flags=cv.CASCADE_SCALE_IMAGE)

    color = (255,0,0)
    for rect in rects:
        cv.rectangle(frame, rect, color, 2)

def detectEye():
    rects = eye_detector.detectMultiScale(gray, 
        scaleFactor=1.1,
        minNeighbors=5, 
        minSize=(30, 30), 
        flags=cv.CASCADE_SCALE_IMAGE)

    color = (0,0,255)
    for rect in rects:
        cv.rectangle(frame, rect, color, 2)


cap = cv.VideoCapture(0)
t0 = time.time()

M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
size = (640, 360)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_s = cv.warpAffine(gray, M, size)

    #frame_s = cv.warpAffine(frame, M, size)

    detectFace()
    detectEye()
    cv.imshow('window', frame)
    t = time.time()
    cv.displayOverlay('window', f'time={t-t0:.3f}')
    t0 = t

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
