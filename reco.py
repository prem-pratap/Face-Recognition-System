#!/usr/bin/env python3
import cv2
import numpy as np
import dlib
import Facerecognize as fr

font=cv2.FONT_ITALIC
cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
face_data,face_id,dic=fr.training_dataset_and_labels()
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_data,np.array(face_id))
while cap.isOpened():
    status,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        value,confidence=recognizer.predict(gray[y1:y2,x1:x2])
        text=dic[str(value)]
        if confidence < 100:
            cv2.putText(frame,text,(x1,y1),font,2,(255,0,0),4)
        #print(value)
        #print(confidence)
    cv2.imshow("live",frame)
    if cv2.waitKey(30) & 0xff==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

