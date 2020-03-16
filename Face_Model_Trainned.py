import numpy as np
import cv2
from os import listdir
from os.path import isfile,join

data_path = 'E:/Python/Attendence_Management _System/Faces/'

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

print(onlyfiles)

Training_Data, Labels = [] ,[]


for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    print(images)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)


Labels = np.asarray(Labels,dtype=np.int32)

#load opencv face detector , I am using LBPH(Local Binary Pattern Histogram) algorithm

model = cv2.face.LBPHFaceRecognizer_create()

#training the model
model.train(np.asarray(Training_Data),np.asarray(Labels))
print("Model Training Complete !!")

face_classifier = cv2.CascadeClassifier('C:/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#function to detect face
def face_detector(img, size = 0.5):
    
    #convert the test image to gray image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detect multiscale images
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    #if no faces are detected then return original img
    if faces is():
        return img,[]

    #extract the face area and return only the face part of the image
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi

#Open an camera
cap = cv2.VideoCapture(0)

while True:
    
    #capture frame-by-frame
    ret, frame = cap.read()
    
    #get two values from face_detector function
    image,face = face_detector(frame)
    try:
        
        #convert the captured face in gray color
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        
        #predict the model
        result = model.predict(face)
        
        #check and calculate the confidence value less than 500
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            
            #display the confidence value on window
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        
        #if confidence value greater than 75% then face matches
        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)

        else:
            
            #if confidence value less than 75% then face not matches
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
    except:
        
        #display the message if face not found, 
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass
    
    if cv2.waitKey(1)==13:
        break
        
#when everthing done, release the capture
cap.release()
cv2.destroyAllWindows()
