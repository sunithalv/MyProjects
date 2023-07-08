# Lane detection in self driving cars
The code detects the lanes on either side of a moving car.The following are the steps:
1) Convert the captured frames in the video to gray scale
2) Use Canny's algorithm for edge detection
3) Get the lower region of interest as a triangular region and generate a mask using the triangular vertices
4) Use bitwise AND operation of original image and mask to get the cropped image
5) Use Hough Algorithm on the cropped image to detect the lines in the image
6) Draw the detected lines on the image

   ![lane_detection](https://github.com/sunithalv/MyProjects/assets/28974154/9452d687-87cc-48bf-bc38-bd27bfe689eb)
