# import cv2
# import numpy as np

# import time
# import sys
# import os

# CONFIDENCE = 0.5
# SCORE_THRESHOLD = 0.5
# IOU_THRESHOLD = 0.5

# # the neural network configuration
# config_path = "cfg/yolov3.cfg"

# print(config_path)
# # the YOLO net weights file
# weights_path = "./yolov5/weights/yolov3.weights"
# print(weights_path)


import cv2
from cv2 import waitKey
import torch
from PIL import Image
import pandas  as pd

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom',path='./yolov5/best.pt')

# Images
# for f in './pic1.jpg', './pic2.jpg':
#     torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
im1 = Image.open('./222.jpg')  # PIL image
# im2 = cv2.imread('./39.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)


# xxxxxxxxxxxxxxxxxxxxxxxxxxx
# cam = cv2.VideoCapture(0)

# while True:
#     check, frame = cam.read()

#     cv2.imshow('video', frame)
#     results = model([frame])
#     results.save()  
#     print(results)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cam.release()
# cv2.destroyAllWindows()
# xxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxx
# Capture video from file


# cap = cv2.VideoCapture(0,)


# while True:

#     ret, frame = cap.read()
#     # cap.set(3,640)
#     # cap.set(4,480)

#     if ret == True:

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

#         cv2.imshow('frame',gray)
#         results = model([gray])
#         results.print()
#         results.save()


#         if cv2.waitKey(30) & 0xFF == ord('q'):
#             break
#     else:
#         break

# waitKey(0)
# cap.release()
# cv2.destroyAllWindows()
# print('CAPPPPPPPPPPPPPPPP')
#  xxxxxxxxxxxxxxxxxxxxxx

# Inference
results = model([im1]) # batch of images

# # Results
results.print()  
# results.save()  # or .show()

# results.xyxy[0]  # im1 predictions (tensor)
# results.pandas().xyxy[0] 

df=results.pandas().xyxy[0]

dfobj =pd.DataFrame(df)

# df_Last_col = dfobj.T.tail(1).T
df_last_col = dfobj.iloc[:, -1].tolist()
 
print('Last column of dataframe:',df_last_col)
print(len(df_last_col))



