#!/usr/bin/env python3
import cv2
import numpy as np
import os
import Facerecognize as fr

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface.xml")
font=cv2.FONT_ITALIC

#calling our trainer
face_data,face_id,dic=fr.training_dataset_and_labels()
#features=np.load('trainingfaces.npy')
#labels=np.load('traininglabel.npy')
#calling LBPH face recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_data,np.array(face_id))
#recognizer.save('trainingdata.yml')
#recognizer.read('trainingdata.yml')
#face_recognizer=f.train_classifier(face_data,face_id)
while cap.isOpened():
    status,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,1.5,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        value,confidence=recognizer.predict(gray[y:y+h,x:x+w])
        text=dic[str(value)]
        #print(value)
        #if confidence < 40:
        cv2.putText(frame,text,(x,y),font,2,(255,0,0),4)
        #print(confidence)
        #print(value)
    cv2.imshow("live",frame)
    if cv2.waitKey(30) & 0xff==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

