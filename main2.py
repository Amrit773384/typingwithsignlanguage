from multiprocessing import Process
from threading import Thread
#gui resources 
import tkinter as tk
from tkinter.ttk import Radiobutton
from tkinter import PhotoImage
from PIL import ImageTk, Image
#gesture resources 
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
# volume control libraries
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np 


class GUI:
    def __init__(self,root):
        self.root = root
        self.root.title("Wireless Sound Control")
        self.root.geometry("600x700")
        self.root.resizable(False,False)

        self.upperframe = tk.Frame(self.root,background='white',height=400,width=600)
        self.upperframe.pack(side='top',fill='x',anchor='n',padx=10,pady=10)
        self.cap = cv2.VideoCapture(1)
        img = self.cap.read()[1]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img).resize((self.upperframe['width'],self.upperframe['height']),Image.Resampling.LANCZOS)) 
        self.camera = tk.Label(self.upperframe,image=img)
        self.camera.pack(side='right')


        self.notepad_frame = tk.Frame(self.root,width=600,height=200) 
        self.notepad_frame.pack(expand=False,side='top')

        self.notepad = tk.Text(self.notepad_frame,font='consolas 14',height=7)
        self.notepad.pack(side='top',padx=10)


        self.bottomframe = tk.Frame(self.root,height=100,width = 600)
        self.bottomframe.pack(fill='both',side='top',padx=10,pady=10)

        commonfont = 'arial 16 '

        self.space_button = tk.Button(self.bottomframe,
                                    text="Space",
                                    font=commonfont,
                                    background='AntiqueWhite1',
                                    padx=20,pady=10,
                                    command=lambda:self.notepad.insert('end', ' ')
                                    )
        self.space_button.grid(row=0,column=0,padx=55,pady=10)
        
        self.clear_button = tk.Button(self.bottomframe,
                                    text='Clear',
                                    font=commonfont,
                                    background='lightgreen',
                                    padx=20,pady=10,
                                    command=lambda:self.notepad.delete(1.0,'end')
                                    )
        self.clear_button.grid(row=0,column=1,pady=10)

        self.savetext_button = tk.Button(self.bottomframe,
                                    text='Save',
                                    font=commonfont,
                                    background='skyblue',
                                    padx=20,pady=10,
                                    command=self.save
                                    )
        self.savetext_button.grid(row=0,column=2,padx=55,pady=10)

        self.notepad.focus_set()

        cap = cv2.VideoCapture(1)

        # Gesture detection variables
        detector = HandDetector(maxHands=1)
        classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        labels = ["A", "B", "C","D","E",'F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',' ']
        offset = 20
        imgSize = 300
        old_alpha = ''
        seconds = 3
        # volumen control variables
        mpHands = mp.solutions.hands 
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volMin,volMax = volume.GetVolumeRange()[:2]

        # Prediction Starting
        seconds = seconds*10
        while cap.isOpened():
            success,img = cap.read()
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            lmList = []
            if results.multi_hand_landmarks:
                for handlandmark in results.multi_hand_landmarks:
                    for id,lm in enumerate(handlandmark.landmark):
                        h,w,_ = img.shape
                        cx,cy = int(lm.x*w),int(lm.y*h)
                        lmList.append([id,cx,cy]) 
                    mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)
            if lmList != []:
                x1,y1 = lmList[4][1],lmList[4][2]
                x2,y2 = lmList[8][1],lmList[8][2]
                cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
                cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)
                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
                length = hypot(x2-x1,y2-y1)
                vol = np.interp(length,[15,220],[volMin,volMax])
                # print(vol,length)
                volume.SetMasterVolumeLevel(vol, None)

            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            try:
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    imgCropShape = imgCrop.shape

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        print(prediction, index)

                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    

                    cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                                (x - offset+90, y - offset-50+50), (0 ,150, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x-offset, y-offset),
                                (x + w+offset, y + h+offset), (0, 150, 255), 4)
            
                    new = labels[index]
                    if old_alpha == new:  
                        out_force += 1
                    elif old_alpha != new: 
                        out_force = 1
                    if out_force>seconds:
                        print("inserting..")
                        self.notepad.insert('end', new)
                        out_force = 1
                        cv2.rectangle(imgOutput, (x-offset, y-offset),
                                (x + w+offset, y + h+offset), (0, 150, 255), -1)
                    old_alpha = new

                cv2image = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image).resize((self.upperframe['width'],self.upperframe['height']),Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                try:
                    self.camera['image'] = imgtk
                    self.root.update()
                except:break
            except cv2.error:
                pass

    def save(self):
            with open('output.txt','w') as file:
                file.write(self.notepad.get(1.0,'end'))
                file.close()
            print(self.notepad.get(1.0, 'end'))
            print("text save ..")
if __name__ == '__main__':

    tk_object = tk.Tk()
    window = GUI(root = tk_object)

    
