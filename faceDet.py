import cv2

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)


faceDataDir = "faceDatas"
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceID = input("Masukkan Face ID yang akan direkam datanya (Kemudian tekan enter): ")
print ("Tatap wajah Anda ke arah webcam. Tunggu proses pengambilan data wajah selesai!")

dataSave = 1
while True:
    retV, frame = cam.read()
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(greyFrame, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (240,252,3), 3)
        fileName = "face."+str(faceID)+"."+str(dataSave)+".jpg"
        cv2.imwrite(faceDataDir+'/'+fileName, frame)
        dataSave += 1

    cv2.imshow("Face Saving", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q') or dataSave > 30: 
        break

cam.release()
cv2.destroyAllWindows()