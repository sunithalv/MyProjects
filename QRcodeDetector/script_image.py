import cv2
import numpy as np

from pyzbar.pyzbar import decode

image = cv2.imread("../Images/MultipleQR_Bar_code.PNG")
for code in decode(image):
    print(code.data.decode('utf-8'))
cv2.imshow("Input Image", image)

cv2.waitKey(0)