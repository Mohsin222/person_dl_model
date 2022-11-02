import cv2
from cv2 import waitKey
import torch
from PIL import Image
import pandas  as pd

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom',path='./yolov5/best.pt')


im1 = Image.open('./222.jpg')


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