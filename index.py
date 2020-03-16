#importing libraries
import tkinter
from tkinter.ttk import *
from tkinter import *
from tkinter import messagebox
import csv,os
import pandas as pd
import numpy as np
import cv2
from os import listdir
from os.path import isfile,join
import datetime
import time
from datetime import date

class screen():
    def __init__(self):
        self.ams=tkinter.Tk()
        self.ams.title("Attendence Management System")

        #Frame:
        self.frame=Frame(self.ams,bg='dark turquoise',width=1600,height=150)
        self.frame.pack(side=TOP)
        
        #Title:
        self.title=Label(self.ams,text='Attendence Management System',fg='white',bg='lightgrey',font=('Arial Black',40))
        self.title.place(x=160,y=32)

        self.fhead=Label(self.ams,text='Attendence',font=('Arial',30),bg='light grey',fg='white')
        self.fhead.place(x=560,y=150)

        #Labels:
        self.CID_Label=Label(self.ams,text='Enter Your ID',bg='black',fg='white',font=('Arial',20))
        self.CID_Label.place(x=150,y=245)

        self.NAME_Label=Label(self.ams,text='Enter Your Name',bg='black',fg='white',font=('Arial',20))
        self.NAME_Label.place(x=542,y=245)

        self.NOTI_Label=Label(self.ams,text='Notification',bg='black',fg='white',font=('Arial',20))
        self.NOTI_Label.place(x=945,y=245)

        self.STEP1_Label=Label(self.ams,text='STEP 1',bg='white',fg='green',font=('Arial',15))
        self.STEP1_Label.place(x=250,y=405)

        self.STEP2_Label=Label(self.ams,text='STEP 2',bg='white',fg='green',font=('Arial',15))
        self.STEP2_Label.place(x=625,y=405)

        self.STEP3_Label=Label(self.ams,text='STEP 3',bg='white',fg='green',font=('Arial',15))
        self.STEP3_Label.place(x=985,y=405)

        self.ATT_Label=Label(self.ams,text='ATTENDENCE',bg='green',fg='white',font=('Arial',30),width=15,height=2)
        self.ATT_Label.place(x=150,y=545)

        self.DISPLAY_Label=Label(self.ams,text='',bg='lightblue',fg='white',font=('Arial',20),width=37,height=3)
        self.DISPLAY_Label.place(x=530,y=545)

        #Entry:
        self.CID_Entry=Entry(self.ams,bd=8,width=18,font=('Arial',17),bg='white',relief='sunken')
        self.CID_Entry.place(x=155,y=305)

        self.NAME_Entry=Entry(self.ams,bd=8,width=18,relief='sunken',bg='white',font=('Arial',16))
        self.NAME_Entry.place(x=535,y=305)

        self.NOTI_Entry=Label(self.ams,bd=8,width=35,relief='sunken',bg='white',font=('Arial',16))
        self.NOTI_Entry.place(x=805,y=305)

        #Button:
        self.IC_But=Button(self.ams,text='IMAGE CAPTURE BUTTON',bd=4,font='8',width=25,relief='ridge',command=self.takesimages)
        self.IC_But.place(x=155,y=455)

        self.MT_But=Button(self.ams,text='MODEL TRAINING BUTTON',bd=4,font='8',width=25,relief='ridge',command=self.trainimages)
        self.MT_But.place(x=535,y=455)

        self.AM_But=Button(self.ams,text='ATTENDENCE MARKING',bd=4,font='8',width=25,relief='ridge',command=self.ImageRecognize)
        self.AM_But.place(x=905,y=455)

        self.Cncl_But=Button(self.ams,text='Cancel',bd=4,font='8',width=10,relief='ridge',command=self.ams.destroy)
        self.Cncl_But.place(x=580,y=670)

        self.ams.geometry("1270x750+0+0")
        self.ams.config(bg='white')
        self.ams.mainloop()

    #this function clear the id text field
    def clear_id(self):
        self.CID_Entry.delete(0,'end')
        self.result = ''
        self.NOTI_Entry.configure(text = self.result)

    #this function clear the name text field
    def clear_name(self):
        self.NAME_Entry.delete(0,'end')
        self.result = ''
        self.NOTI_Entry.configure(text = self.result)

    #this function collecting the images samples
    def takesimages(self):
        self.Id = self.CID_Entry.get()
        self.name = self.NAME_Entry.get()
        
        #check if id text field  is filled or not 
        if not self.Id:
            #showerror('Warning','Please enter ID')
            self.result = "Please enter ID"
            self.NOTI_Entry.configure(text = self.result)
        
        #check if name text field  is filled or not
        elif not self.name:
            #showerror("Warning","Please enter Name")
            self.result = "Please enter Name"
            self.NOTI_Entry.configure(text = self.result)
        
        #collecting images samples
        elif(self.Id.isnumeric() and self.name.isalpha()):
            self.clear_id()
            self.clear_name()
            #capture frames from a camera
            self.cap = cv2.VideoCapture(0)

            #load cascadeclassifier
            self.detector = cv2.CascadeClassifier('C:/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
            self.count = 0

            #loop runs if capturing has been initialized. 
            while(True):

                #read frames from a camera 
                self.ret, self.img = self.cap.read()
                #convert to gray scale of each frames
                self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                #detect multiscale images
                self.faces = self.detector.detectMultiScale(self.gray, 1.3, 5)
                
                #extract the face area and return only the face part of the image
                for (x,y,w,h) in self.faces:
                    cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),2) 
                    #incrementing sample number 
                    self.count = self.count + 1
                    #saving the captured face in the dataset folder Faces
                    cv2.imwrite("Faces\ "+self.name +"."+self.Id +'.'+ str(self.count) + ".jpg", self.gray[y:y+h,x:x+w])
                    #display the frame
                    cv2.imshow('frame',self.img)
                #wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == 13:
                    break
                # break if the sample number is more than 60
                elif self.count > 60:
                    break
            #when everthing done, release the capture
            self.cap.release()
            cv2.destroyAllWindows()

            #display notification 
            self.result = "Images Saved for ID : " + self.Id +" Name : "+ self.name
            self.row = [self.Id , self.name]
            
            #open a csv file in append mode
            with open('CSV\SaveDetails.csv','a+') as self.csvFile:
                self.w = csv.writer(self.csvFile)
                self.w.writerow(self.row)
            
            #close cvs file
            self.csvFile.close()
            self.NOTI_Entry.configure(text = self.result)
        else:
            #check if id field is numeric or not
            if not self.Id.isnumeric():
                self.result = "Enter Numeric Id"
                self.NOTI_Entry.configure(text = self.result)
            
            #check if name field is alphabetical or not
            if not self.name.isalpha():
                self.result = "Enter Alphabetical Name"
                self.NOTI_Entry.configure(text = self.result)

    #function to train model
    def trainimages(self):
        #load opencv face detector , I am using LBPH(Local Binary Pattern Histogram) algorithm
        self.model = cv2.face.LBPHFaceRecognizer_create()

        #getting training data and labels from getImagesandLabels() function
        self.Training_Data,self.Labels = self.getImagesandLabels("Faces")

        #training a model
        self.model.train(np.asarray(self.Training_Data),np.asarray(self.Labels))

        #save a model
        self.model.save("ModelSaving\Trainner.yml")
        self.clear_id()
        self.clear_name()

        #display notification
        self.result = "Image Trained"
        self.NOTI_Entry.configure(text = self.result)
    
    #function to getting images and labels
    def getImagesandLabels(self,data_path):
        self.data_path = 'E:/Python/Attendence_Management _System/Faces/'

        #file name of all images
        self.onlyfiles = [f for f in listdir(self.data_path) if isfile(join(self.data_path,f))]
        
        #create empty list
        self.Training_Data, self.Labels = [] ,[]
        
        #getting images and labels
        for i, files in enumerate(self.onlyfiles):
            self.image_path = self.data_path + self.onlyfiles[i]
            
            #convert the image in grayscale mode 
            self.images = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)

            #convert images into numpy array and append in training data
            self.Training_Data.append(np.asarray(self.images,dtype=np.uint8))

            #getting only id number from image name
            self.Id = int(os.path.split(files)[-1].split(".")[1])

            #appending ids in labels
            self.Labels.append(self.Id)
        
        #convert the labels into numpy array
        self.Labels = np.asarray(self.Labels,dtype=np.int32)
        return self.Training_Data,self.Labels

    #function to face recognition
    def ImageRecognize(self):
        #load opencv face detector , I am using LBPH(Local Binary Pattern Histogram) algorithm
        self.r = cv2.face.LBPHFaceRecognizer_create()
        
        #read a saved model file
        self.r.read('ModelSaving\Trainner.yml')

        #load opencv face detector , I am using LBPH(Local Binary Pattern Histogram) algorithm
        self.faceCascade = cv2.CascadeClassifier('C:/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        
        #read dataframe file
        self.df = pd.read_csv('CSV\SaveDetails.csv')
        
        #Open an camera
        self.cap = cv2.VideoCapture(0)

        #font style
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        #column names of dataframe
        self.col_names = ['Id','Name','Date','Time']

        #creating a new dataframe
        self.df_attendence = pd.DataFrame(columns = self.col_names)

        #loop runs if capturing has been initialized
        while True:
            #capture frame-by-frame
            self.ret, self.im = self.cap.read()
            #convert the captured face in gray color
            self.gray = cv2.cvtColor(self.im,cv2.COLOR_BGR2GRAY)
            self.faces = self.faceCascade.detectMultiScale(self.gray, 1.2,5)

            #extract the face area and return only the face part of the image
            for(x,y,w,h) in self.faces:
                cv2.rectangle(self.im,(x,y),(x+w,y+h),(225,0,0),2)

                #predict the model
                self.Id, self.conf = self.r.predict(self.gray[y:y+h,x:x+w])

                #if confidence value smaller than 50% then face matches                        
                if(self.conf < 50):
                    self.t = datetime.datetime.now()
                    self.date = self.t.strftime("%x")
                    self.timeStamp = self.t.strftime("%X")
                    #getting the name
                    self.aa = self.df.loc[self.df['Id'] == self.Id]['Name'].values
                    #getting the id with name
                    self.tt = str(self.Id) + "-" + self.aa
                    #return id, name, date, time
                    self.df_attendence.loc[len(self.df_attendence)] = [self.Id,self.aa,self.date,self.timeStamp]
                    
                else:
                    self.Id = 'Unknown'     
                    self.tt = str(self.Id) 
               
                #if confidence value greater than 75% then face not matches 
                if(self.conf > 75):
                    self.noOfFile = len(os.listdir("ImagesUnknown"))+1

                    #saving unknown images in ImagesUnknown folder
                    cv2.imwrite("ImagesUnknown\Image"+str(self.noOfFile) + ".jpg", self.im[y:y+h,x:x+w])            
                cv2.putText(self.im,str(self.tt),(x,y+h), self.font, 1,(255,255,255),2)        
            
            #returns a single id
            self.df_attendence = self.df_attendence.drop_duplicates(subset=['Id'],keep='first')
            
            #display a face in a window
            cv2.imshow('im',self.im) 
            
            #Wait for Enter key to stop
            if (cv2.waitKey(1) ==  13):
                break
        
        #date and time
        self.t = datetime.datetime.now()
        self.date = self.t.strftime("%x")
        self.dt = self.t.strftime("%d-%m-%Y")
        self.timeStamp = self.t.strftime("%X")      
        self.Hour,self.Minute,self.Second = self.timeStamp.split(":")

        #update attendence in Attendence folder
        self.fileName = "Attendance\\Attendance_" + self.dt + "-" + self.Hour + "-" + self.Minute +"-" + self.Second +".csv"
        self.df_attendence.to_csv(self.fileName,index=False)
        
        #when everthing done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()
        
        #display notification
        self.result = self.df_attendence
        self.DISPLAY_Label.configure(text= self.result)
        self.result = "Attendance Taken"
        self.NOTI_Entry.configure(text= self.result)


w=screen()