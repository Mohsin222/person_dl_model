import cv2
from cv2 import waitKey
import torch
from PIL import Image
import pandas  as pd

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom',path='./yolov5/best.pt')




# xxxxxxxxxxxxxxxxxxxxxxxx
# Capture video from file


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()
    # cap.set(3,640)
    # cap.set(4,480)

    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        cv2.imshow('frame',gray)
        results = model([gray])
        # results.print()
        # results.save()
        df=results.pandas().xyxy[0]

        dfobj =pd.DataFrame(df)
        

# df_Last_col = dfobj.T.tail(1).T
        df_last_col = dfobj.iloc[:, -1].tolist()
        print('Last column of dataframe:',df_last_col)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

waitKey(0)
cap.release()
cv2.destroyAllWindows()
print('CAPPPPPPPPPPPPPPPP')
#  xxxxxxxxxxxxxxxxxxxxxx
