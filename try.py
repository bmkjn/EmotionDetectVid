import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np 
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import imutils

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

class EmotionDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('800x600')
        self.root.title('Emotion Detector')
        self.root.configure(background='#CDCDCD')
        

        self.label1 = Label(self.root, background='#CDCDCD', font=('arial',15,'bold'))
        self.sign_image = Label(self.root)

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        

        self.model = FacialExpressionModel("model_a1.json", "model_weights1.h5")

        self.EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        self.cap = cv2.VideoCapture(0)  
        

        self.upload_button = Button(self.root, text="Start Emotion Detection", command=self.start_detection, padx=10, pady=5)
        self.upload_button.configure(background="#364156", foreground="white", font=("arial", 20, "bold"))
        self.upload_button.pack(side='bottom', pady=50)
        

        self.sign_image.pack(side='bottom', expand='True')
        self.label1.pack(side='bottom', expand='True')
        self.heading = Label(self.root, text="Emotion Detector", pady=20, font=("arial", 25, "bold"))
        self.heading.configure(background="#CDCDCD", foreground="#364156")
        self.heading.pack()
        


    def start_detection(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=450)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi = gray_frame[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                pred = self.EMOTIONS_LIST[np.argmax(self.model.predict(roi[np.newaxis, :, :, np.newaxis]))]

            cv2.putText(frame, "**************** "+pred+" ****************", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "**************** "+pred+" ****************", (10,325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




            print("Predicted Emotion is " + pred)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = EmotionDetector()
    app.run()

