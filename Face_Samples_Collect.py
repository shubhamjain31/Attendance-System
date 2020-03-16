import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('C:/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    
    #Convert to gray scale of each frames
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #Detects faces of different sizes in the input image 
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    
    # To draw a rectangle in a face
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w]
        
    return cropped_face


# capture frames from a camera 

cap = cv2.VideoCapture(0)
count = 0


# loop runs if capturing has been initialized. 

while True:
    
    # reads frames from a camera 
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        
        # convert to gray scale of each frames
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        
        file_name_path = 'E:/Python/Attendence_Management _System/Faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
        #Display an face in a window
        cv2.imshow('Face Cropper',face)
    else:
        print('Face not found')
        pass
    
    #Wait for Esc key to stop
    if cv2.waitKey(1) == 13 or count == 100:
        break

#Close the window         
cap.release()

#De-allocate any associated memory usage 
cv2.destroyAllWindows()
print('Collecting Samples Complete !!')
