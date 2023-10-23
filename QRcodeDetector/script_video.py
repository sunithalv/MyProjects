import cv2
import numpy as np
from pyzbar.pyzbar import decode
#Authorize barcodes based on whether the decoded data is in the list of values provided in a text file
cap=cv2.VideoCapture('../Videos/QRVideo2.mp4')
#get the list of approved data
with open("../Resources//myData.txt") as f:
    myDataList = f.read().splitlines()
    print("My Approved Data", myDataList)

count=0
while True:
    ret, frame = cap.read()
    if ret:
        #print(f"Frame Count {count}")
        frame = cv2.resize(frame, (0,0), None, 0.7,0.7)
        for code in decode(frame):
            mydata = code.data.decode('utf-8')
            #print("Decoded Data", mydata)
            if mydata in myDataList:
                myOutut = "Authorized"
                color_value=(0,255,0)
            else:
                myOutut = "Un-Authorized"
                color_value=(0,0,255)
            #Polygon data is available in the decoded data
            pts = np.array([code.polygon], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 5)
            pts2 = code.rect
            cv2.putText(frame, myOutut, (pts2[0], pts2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_value,2)
        count+=1
        cv2.imshow("Input Video", frame)

        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    else:
        break