import cv2, os, numpy as np

faceDataDir = "faceDatas"
learnDir = "learning"
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)


faceRecognizer.read(learnDir+'/trained.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Unknown', 'Irwan Fuadi', 'Mas Ipong']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(greyFrame, 1.2, 5, minSize = (round(minWidth), round(minHeight)), )
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (240,252,3), 3)
        id, confidence = faceRecognizer.predict(greyFrame[y:y+h,x:x+h])

        if  confidence < 60:
            nameID = names[id]
            confidenceTxt = " {0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTxt = " {0}%".format(round(100-confidence))
        
        cv2.putText(frame, str(nameID), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidenceTxt), (x+5,y+h-5), font, 1, (255,255,255), 1)

    cv2.imshow("Face Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'): 
        break

cam.release()
cv2.destroyAllWindows()


