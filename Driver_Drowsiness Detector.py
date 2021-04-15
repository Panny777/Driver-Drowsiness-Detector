import dlib
from imutils import face_utils
import cv2
import numpy as np
from scipy.spatial import distance as dist
import threading
import pygame
from playsound import playsound
import time
from datetime import datetime
import pyttsx3


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
rate = engine.getProperty('rate')
volume = engine.getProperty('volume')
engine.setProperty('voice', voices[2].id)

def speak(audio):
    print('Jovial:', audio)
    newVoiceRate = 160
    engine.setProperty('rate',newVoiceRate)
    engine.say(audio)
    engine.runAndWait()

def alert_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("sounds/alert.mp3")
    pygame.mixer.music.play()

url = "http://192.168.1.34:8080/video"

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36,48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# A function of getting the aspect ratio of the eye//// the size and shape
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
   
	# compute the eye aspect ratio
    eye = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
    return eye
camera = cv2.VideoCapture(url)

predictor_path = (R"C:\Users\muhid\OneDrive\Documents\Python Projects\drowsinessDetector\shape_predictor_68_face_landmarks.dat_2")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total=0
alarm=False
while True:
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera.\n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=60)

# Ask the detector to find the bounding boxes of each face. The 1 in the
    dets = detector(frame_resized, 1)
    status = 'Status: '
    aspect_ratio = 'Aspect Ratio: '
    cv2.putText(frame, status, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, aspect_ratio, (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)
    
    
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            leftEye= shape[lStart:lEnd]
            rightEye= shape[rStart:rEnd]
            leftEAR= eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            eye = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
	       
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            eye = round(eye,2)
            asp_ratio = "Aspect Ratio: " + str(eye)
            if eye>.25:
                eye = round(eye,2)
                total=0
                alarm=False
                cv2.putText(frame, status + "Eyes Opened", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, asp_ratio, (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)
            else:
                total+=1
                if total>20:
                    if not alarm:
                        cv2.putText(frame, "Drowsy Detected" ,(250, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
                        # if the driver has closed the eyes for the given seconds the alert will go off
                        time.sleep(10)
                        alarm = True
                        d=threading.Thread(target=alert_sound)
                        d.setDaemon(True)
                        d.start()
                cv2.putText(frame, status + "Eyes closed".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, asp_ratio, (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)

            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (0, 255, 0), -1)
    cv2.resizeWindow("Driver Drowsiness Detector", 1366,768)
    cv2.imshow("Driver Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
