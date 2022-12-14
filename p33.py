from turtle import shape
import cv2
import torch
from PIL import Image
import pandas  as pd



cap = cv2.VideoCapture('vid1.mp4',shape)
def rescale_frame(frame, percent=30):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA) 
if (cap.isOpened() == False):
    print("Error opening video stream or file") 
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read() 
frame = rescale_frame(frame, percent=30)
if ret == True:
    cv2.imshow('Frame', frame)

if cv2.waitKey(25) & 0xFF == ord('q'):
  


    
    

cv2.destroyAllWindows()