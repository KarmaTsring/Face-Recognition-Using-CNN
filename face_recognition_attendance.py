import cv2
import face_recognition
import numpy as np
import os
import datetime as dt

#path to the images
path = 'Images'

images = []
peopleNames = []

myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')

    images.append(curImg)

    peopleNames.append(os.path.splitext(cl)[0])

print(peopleNames)

"Encodingfaces"
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList




encodeListKnown = findEncodings(images)
print('Encoding completed')

#webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSml = cv2.resize(img,(0,0), None, 1,1)
    imgSml = cv2.cvtColor(imgSml, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSml)
    encodeCurFrame = face_recognition.face_encodings(imgSml,facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)

        #lowest match index
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = peopleNames[matchIndex].upper()
            print(name)



            #due to faceLoc rule of faceLoc(height,top,bottom,left)
            y,w,h,x = faceLoc
            #y, w, h, x = y*2, w*2, h*2, x*2
            cv2.rectangle(img,(x,y),(w,h),(120,120,0),1)
            cv2.rectangle(img,(x,h-25),(w,h),(225,225,225),cv2.FILLED)
            cv2.putText(img,name,(x+6,h-6),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)