#!/usr/bin/env python3
import cv2
import os

count=0
cap=cv2.VideoCapture(0)
#import haar file
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface.xml")
name=str(input("Enter your name:"))
sid=str(input("Enter your id:"))

dir_name="/home/prem/Desktop/summer/dataset_images/"+sid
try:
    os.mkdir(dir_name)
except:
    print("Directory already exist")

    
while cap.isOpened():
    status,frame=cap.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_img,1.15,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        count=count+1
        #print(count)
        cv2.imwrite(dir_name+"/"+name+"_"+str(count)+".jpg",gray_img)
    cv2.imshow("live",frame)
    if (cv2.waitKey(50) & 0xff==ord('q')) | count==50:
        break
cv2.destroyAllWindows()
cap.release()

	
