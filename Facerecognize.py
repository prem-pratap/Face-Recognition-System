#!/usr/bin/env python3
import numpy as np
import cv2
import os

def training_dataset_and_labels():
    #import haar file
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface.xml")
    dir_name="/home/prem/Desktop/summer/dataset_images/"
    #creating features and labels
    face_data=[]
    labels=[]
    count=0
    subdir=os.listdir(dir_name)
    dic={x: 0 for x in subdir}
    for i in subdir:
        folder_path=dir_name+str(i)
        images=os.listdir(folder_path)
        for j in images:
            image_path=folder_path+"/"+j 
            face_values=cv2.imread(image_path,0)
            #face_detection in read images
            faces=face_cascade.detectMultiScale(face_values,1.15,5)
            for (x,y,w,h) in faces:
                face_data.append(face_values[y:y+h,x:x+w])
                part=j.split("_")[-2]
                labels.append(int(i))
            for k in dic.keys():
                part=j.split("_")[-2]
                dic[k]=part
    np.save('trainingfaces',face_data)
    np.save('traininglabel',labels)
    #print(face_data)
    #print(dic)
    #print(labels)
    return face_data,labels,dic



