import cv2
import tkinter as tk
import tkinter.messagebox as msgbox
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image

from datetime import datetime
import sys, os
cwd = os.getcwd()
sys.path.append(cwd)

import requests
from io import BytesIO
import base64

class App(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.subQuestion = tk.StringVar()
        self.subAns = tk.StringVar()
        self.imgPath = ""
        self.camOpened = False
        self.cam = None
        self.stream = None

        self.form = tk.Frame(self, bd = 10)

        self.imgLbl = tk.Label(self.form)
        self.imgLbl.pack()

        self.openCamBtn = tk.Button(
            self.form, 
            text = 'OPEN CAMERA', 
            command = self.onOpenCam
        )
        self.openCamBtn.pack(fill = 'x', pady = [20, 5])

        self.captureImgBtn = tk.Button(
            self.form, 
            text = 'CAPTURE IMAGE', 
            command = self.onCaptureImg
        )
        self.captureImgBtn.pack(fill = 'x', pady = [20, 5])

        self.uploadBtn = tk.Button(
            self.form, 
            text = 'UPLOAD IMAGE', 
            command = self.onUploadImg
        )
        self.uploadBtn.pack(fill = 'x', pady = 5)

        self.questionLbl = tk.Label(
            self.form, 
            text = 'Question', 
        )
        self.questionLbl.pack(anchor = 'w', pady = 5)

        self.questionEntry = tk.Entry(
            self.form, 
            textvariable = self.subQuestion,
            width = 100
        )
        self.questionEntry.pack(fill = 'x', ipady = 5)

        self.ansLbl = tk.Label(
            self.form, 
            text = 'Answer', 
        )
        self.ansLbl.pack(anchor = 'w', pady = 5)

        self.ansEntry = tk.Entry(
            self.form, 
            textvariable = self.subAns,
            state = 'readonly'
        )
        self.ansEntry.pack(fill = 'x', ipady = 5)

        self.subBtn = tk.Button(
            self.form, 
            text = 'SUBMIT', 
            command = self.onSubmit
        )
        self.subBtn.pack(fill = 'x', pady = [20, 5])
            
        self.form.pack(fill = 'both')

    def onOpenCam(self):
        if self.camOpened and self.stream:
            return
        if not self.camOpened:
            self.cam = cv2.VideoCapture(0)
            if not self.cam.isOpened():
                msgbox.showerror('Error', 'Cannot open camera')
                return

        self.camOpened = True
        self.playVidStream()

    def playVidStream(self):
        _, frame = self.cam.read()
        scale_percent = 60 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height) 
        frame = cv2.resize(frame, dim)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image = img)
        self.imgLbl.imgtk = imgtk
        self.imgLbl.configure(image=imgtk)
        self.stream = self.imgLbl.after(1, self.playVidStream) 

    def onCaptureImg(self):

        if self.stream:
            # capture image
            _, frame = self.cam.read()
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            # stop video stream
            self.camOpened = False
            if self.cam:
                del(self.cam)
            self.imgLbl.after_cancel(self.stream)
            self.stream = None
   
            # save captured image into image log directory
            img_log_dir_path = os.path.join("imglog")
            if not os.path.exists(img_log_dir_path):
                os.makedirs(img_log_dir_path)
            img_filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".jpg"
            self.imgPath = img_log_dir_path + "/" + img_filename
            img.save(self.imgPath)

        else:
            msgbox.showerror('Error', 'Camera not opened')
            return


    def onUploadImg(self):
        chosenFile = askopenfilename()
        #filetypes=[("IMAGE FILE", "*.jpg"),("IMAGE FILE", "*.JPG")]
        # if image file found
        if chosenFile:
            
            # stop video stream
            self.camOpened = False
            if self.cam:
                del(self.cam)
            if self.stream:
                self.imgLbl.after_cancel(self.stream)
                self.stream = None

            # read image from file
            img = Image.open(chosenFile)
            self.imgPath = chosenFile
            img = img.resize((200,200))
            imgtk = ImageTk.PhotoImage(image = img)
            self.imgLbl.imgtk = imgtk
            self.imgLbl.configure(image=imgtk)


    def onSubmit(self):
        # check if image path and question are given
        if self.imgPath != "" and self.subQuestion.get() != "":
            img = Image.open(self.imgPath).convert('RGB')
            buff = BytesIO()            
            img.save(buff,"JPEG")
            img_base64 = base64.b64encode(buff.getvalue()).decode()
            
            inputs = {'image': img_base64, 'question': self.subQuestion.get()}
            url = 'http://192.168.1.60:8080'
            resp = requests.post(url, json=inputs)
            data = resp.json()
            answer = data['answer']
            print("ANS:" + answer)
            self.subAns.set(answer)
        else:
            msgbox.showerror('Error', 'No image or empty question')
            return

if __name__ == '__main__':

    # Create application window

    window = tk.Tk()
    window.title('GUI')

    # Setup application frame

    app = App(window)
    app.pack()

    # Start application

    window.mainloop()
