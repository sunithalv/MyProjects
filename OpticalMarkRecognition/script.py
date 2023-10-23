import cv2
import numpy as np

image= cv2.imread("Images/my4.PNG")
width = 700
height = 700
questions = 5
choices = 5
#List of correct answers
ans = [2,2,0,1,2]
image = cv2.resize(image, (width,height))
original_image_copy = image.copy()
original_image_copy_two = image.copy()
def preprocessing_image(image):
    # Convert the Image to Gray Scale
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(gray_scale_image, (5,5), 1)
    #Apply Canny Edge Detector
    lower_threshold = 10
    higher_threshold = 70
    canny_edge = cv2.Canny(gaussian_blur, lower_threshold, higher_threshold)

    return canny_edge

def get_contours(preprocessed_image, output_image, minArea=1000):
    contours, hirearchy = cv2.findContours(preprocessed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Length of the Contours", len(contours))
    rectCon=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("Area of each of the Contour", area)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx)==4:
                rectCon.append(cnt)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    peri1 = cv2.arcLength(rectCon[0], True)
    #Corner points for the rectangle containing the bubbles
    approx1=cv2.approxPolyDP(rectCon[0], 0.02*peri, True)
    peri2 = cv2.arcLength(rectCon[1], True)
    #Corner points for the rectangle for grade
    approx2=cv2.approxPolyDP(rectCon[1], 0.02*peri, True)
    approx1 = reorder(approx1)
    approx2 = reorder(approx2)

    cv2.drawContours(output_image, approx1, -1, (255,0,0), 30)
    cv2.drawContours(output_image, approx2, -1, (255,0,0), 30)
    return output_image, approx1, approx2
def warp_perscpective(image, approx1, approx2):
    #For the rectabgle containing the bubbles
    pts1 = np.float32(approx1)
    pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_warped_bubble = cv2.warpPerspective(image, matrix, (width, height))

    #For the grade rectangle
    ptsG1 = np.float32(approx2)
    ptsG2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(ptsG1, ptsG2)
    image_warped_grade = cv2.warpPerspective(image, matrix, (width, height))

    return image_warped_bubble, image_warped_grade, pts1, pts2, ptsG1, ptsG2

def apply_threshold_bubble(image_warped_bubble):
    image_warped_bubble_grayscale = cv2.cvtColor(image_warped_bubble, cv2.COLOR_BGR2GRAY)
    image_warped_bubble_threshold = cv2.threshold(image_warped_bubble_grayscale, 170, 255, cv2.THRESH_BINARY_INV)[1]
    return image_warped_bubble_threshold

#Split the bubbles into rows and columns
def split_boxes(image):
    #vertical split into 5 rows
    row = np.vsplit(image, 5)
    boxes = []
    for r in row:
        #Horizontal split into 5 columns
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes

#Identify marked and unmarked bubbles
def count_non_zero(boxes):
    #Count rows
    countR = 0
    #Count columns
    countC = 0
    #Total no of quesns is 5 and tot no of choices is 5
    myPixelVal = np.zeros((questions, choices))
    #Iterate each bubble and save nonzero pixel values in 2D array
    for image in boxes:
        #For marked bubbles nonzero pixel value will be high
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC+=1
        if (countC==choices):
            countC = 0
            countR+=1
    print("Non Zero Pixel Values", myPixelVal)
    return myPixelVal

#Get indices of marked bubbles
def index_values_marked_bubble(myPixelVal, questions):
    myIndex = []
    for x in range(0, questions):
        arr= myPixelVal[x]
        myIndexVal = np.where(arr==np.max(arr))
        myIndex.append(myIndexVal[0][0])
    print("My Index Value", myIndex)
    return myIndex

# Compare the marked values with the Original Answers to find the Correct Answers
def compare_values(questions, myIndex):
    grading = []
    for x in range(0, questions):
        #If correct answer add 1 to grading else 0
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    #Get percentage score
    score=(sum(grading)/questions)*100
    print("My Score", score)
    return score, grading

def showanswers(image_warped_bubble, myIndex, grading, ans, questions, choices):
    secW = int(image_warped_bubble.shape[1]/questions)
    secH = int(image_warped_bubble.shape[0]/choices)
    for x in range(0, questions):
        myAns = myIndex[x]
        #Centre points cx,cy
        cx = (myAns*secW) + secW // 2
        cy = (x * secH) + secH // 2
        if grading[x] == 1:
            #Green colour for correct answer
            myColor = (0,255,0)
            cv2.circle(image_warped_bubble, (cx, cy), 50, myColor, cv2.FILLED)
        else:
            #Red colour for wrong answer
            myColor = (0,0, 255)
            cv2.circle(image_warped_bubble, (cx, cy), 50, myColor, cv2.FILLED)

            # Display Correct Answer in case of the wrong answer
            myColor = (0,255,0)
            correctAns = ans[x]
            cv2.circle(image_warped_bubble, ((correctAns * secW)+secW//2, (x*secH)+secH//2), 50, myColor, cv2.FILLED)
    return image_warped_bubble

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew
preprocessed_image = preprocessing_image(image)
draw_contours, approx1, approx2 = get_contours(preprocessed_image, original_image_copy, minArea=1000)

image_warped_bubble, image_warped_grade, pts1, pts2, ptsG1, ptsG2 = warp_perscpective(image, approx1, approx2)

image_warped_bubble_threshold = apply_threshold_bubble(image_warped_bubble)

boxes = split_boxes(image_warped_bubble_threshold)
myPixelVal=count_non_zero(boxes)
index_values = index_values_marked_bubble(myPixelVal, questions)
score, grading = compare_values(questions, index_values)
#To show the correct(green) and wrong(red) answers in the warped image
image_warped_bubble = showanswers(image_warped_bubble, index_values, grading, ans, questions, choices)


image_warped_bubble_black = np.zeros_like(image_warped_bubble)
#To show the correct and wrong answers in black background
image_warped_bubble_black = showanswers(image_warped_bubble_black, index_values, grading, ans, questions, choices)

#Get the inverse warp matrix
invmatrix_bubles = cv2.getPerspectiveTransform(pts2, pts1)
#Get the inverse warp to get original image with the green and red markings
image_bubble_inv_warp = cv2.warpPerspective(image_warped_bubble_black, invmatrix_bubles, (width, height))

#Add the two images
imageFinal = cv2.addWeighted(original_image_copy_two, 1, image_bubble_inv_warp, 1, 0)

#To display the grade in the rectangle on the image
image_warped_grade_black = np.zeros_like(image_warped_grade, np.uint8)

#Add the score to the image
cv2.putText(image_warped_grade_black, str(int(score)) + "%", (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 3)

#get inverse matrix and warp for grade
invmatrix_grade = cv2.getPerspectiveTransform(ptsG2, ptsG1)
image_grade_inv_warp = cv2.warpPerspective(image_warped_grade_black, invmatrix_grade, (width, height))

#Add the two images to get the grade in the original image
imageFinal = cv2.addWeighted(imageFinal, 1, image_grade_inv_warp, 1, 0)


#print(cv2.countNonZero(boxes[0]), cv2.countNonZero(boxes[1]))
#cv2.imshow("Input Image", image)
#cv2.imshow("Preprocessed Image", preprocessed_image)
#cv2.imshow("Drawing the Contours", draw_contours)

#cv2.imshow("Warp Perscpective Bubble Area", image_warped_bubble)

#cv2.imshow("Warp Perspective Grade Area", image_warped_grade)

#cv2.imshow("Appling Threshold on Bubble Area", image_warped_bubble_threshold)
#cv2.imshow("1st box", boxes[0])
#cv2.imshow("Show Answers on Warped Image", image_warped_bubble)
#cv2.imshow("Blank Image", image_warped_bubble_black)
cv2.imshow("Final Image", imageFinal)
cv2.waitKey(0)