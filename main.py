import cv2

cv2.namedWindow("testing")
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("testing", frame)
    rval, frame = vc.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    key = cv2.waitKey(20)
    if key == 27 or vc.isClosed(): # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")


emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
