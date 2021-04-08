import cv2, os, numpy as np
from PIL import Image

faceDataDir = "faceDatas"
learnDir = "learning"
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []

    for imagePath in imagePaths:
        pillowImage = Image.open(imagePath).convert('L')
        imgNum = np.array(pillowImage, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)

        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceID)
        return faceSamples, faceIDs

print ("Machine is proccessing! Wait up until it's done.")
faces, IDs = getImageLabel(faceDataDir)
faceRecognizer.train(faces, np.array(IDs))

#SAVE
faceRecognizer.write(learnDir+'/trained.xml')
print ("faces learned by machine!", format(len(np.unique(IDs))))