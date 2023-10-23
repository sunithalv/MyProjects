import cv2
import numpy as np

#image= cv2.imread("Images/my4.PNG")
width = 700
height = 700
questions = 5
choices = 5
ans = [2,2,2,1,2]
count = 0
webcam = False

#image = cv2.resize(image, (width,height))
#original_image_copy = image.copy()
#original_image_copy_two = image.copy()
def preprocessing_image(image):
    # Convert the Image to Gray Scale
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blurr
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
    approx1=cv2.approxPolyDP(rectCon[0], 0.02*peri, True)
    peri2 = cv2.arcLength(rectCon[1], True)
    approx2=cv2.approxPolyDP(rectCon[1], 0.02*peri, True)
    approx1 = reorder(approx1)
    approx2 = reorder(approx2)

    cv2.drawContours(output_image, approx1, -1, (255,0,0), 30)
    cv2.drawContours(output_image, approx2, -1, (255,0,0), 30)
    return output_image, approx1, approx2
def warp_perscpective(image, approx1, approx2):
    pts1 = np.float32(approx1)
    pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_warped_bubble = cv2.warpPerspective(image, matrix, (width, height))

    ptsG1 = np.float32(approx2)
    ptsG2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(ptsG1, ptsG2)
    image_warped_grade = cv2.warpPerspective(image, matrix, (width, height))

    return image_warped_bubble, image_warped_grade, pts1, pts2, ptsG1, ptsG2

def apply_threshold_bubble(image_warped_bubble):
    image_warped_bubble_grayscale = cv2.cvtColor(image_warped_bubble, cv2.COLOR_BGR2GRAY)
    image_warped_bubble_threshold = cv2.threshold(image_warped_bubble_grayscale, 170, 255, cv2.THRESH_BINARY_INV)[1]
    return image_warped_bubble_threshold

def split_boxes(image):
    row = np.vsplit(image, 5)
    boxes = []
    for r in row:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes
def count_non_zero(boxes):
    countR = 0
    countC = 0
    myPixelVal = np.zeros((questions, choices))
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC+=1
        if (countC==choices):
            countC = 0
            countR+=1
    print("Non Zero Pixel Values", myPixelVal)
    return myPixelVal

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
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)

    score=(sum(grading)/questions)*100
    print("My Score", score)
    return score, grading

def showanswers(image_warped_bubble, myIndex, grading, ans, questions, choices):
    secW = int(image_warped_bubble.shape[1]/questions)
    secH = int(image_warped_bubble.shape[0]/choices)
    for x in range(0, questions):
        myAns = myIndex[x]
        cx = (myAns*secW) + secW // 2
        cy = (x * secH) + secH // 2
        if grading[x] == 1:
            myColor = (0,255,0)
            cv2.circle(image_warped_bubble, (cx, cy), 50, myColor, cv2.FILLED)
        else:
            myColor = (0,0, 255)
            cv2.circle(image_warped_bubble, (cx, cy), 50, myColor, cv2.FILLED)

            # Correct Answer
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
while True:
    if webcam:
        ret, frame = cap.read()
    else:
        image = cv2.imread("Images/my5.PNG")
    image = cv2.resize(image, (width, height))
    original_image_copy = image.copy()
    original_image_copy_two = image.copy()
    try:
        preprocessed_image = preprocessing_image(image)
        draw_contours, approx1, approx2 = get_contours(preprocessed_image, original_image_copy, minArea=1000)

        image_warped_bubble, image_warped_grade, pts1, pts2, ptsG1, ptsG2 = warp_perscpective(image, approx1, approx2)

        image_warped_bubble_threshold = apply_threshold_bubble(image_warped_bubble)

        boxes = split_boxes(image_warped_bubble_threshold)
        myPixelVal=count_non_zero(boxes)
        index_values = index_values_marked_bubble(myPixelVal, questions)
        score, grading = compare_values(questions, index_values)
        image_warped_bubble = showanswers(image_warped_bubble, index_values, grading, ans, questions, choices)

        image_warped_bubble_black = np.zeros_like(image_warped_bubble)

        image_warped_bubble_black = showanswers(image_warped_bubble_black, index_values, grading, ans, questions, choices)

        invmatrix_bubles = cv2.getPerspectiveTransform(pts2, pts1)
        image_bubble_inv_warp = cv2.warpPerspective(image_warped_bubble_black, invmatrix_bubles, (width, height))

        imageFinal = cv2.addWeighted(original_image_copy_two, 1, image_bubble_inv_warp, 1, 0)

        image_warped_grade_black = np.zeros_like(image_warped_grade, np.uint8)

        cv2.putText(image_warped_grade_black, str(int(score)) + "%", (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 3)

        invmatrix_grade = cv2.getPerspectiveTransform(ptsG2, ptsG1)

        image_grade_inv_warp = cv2.warpPerspective(image_warped_grade_black, invmatrix_grade, (width, height))
        imageFinal = cv2.addWeighted(imageFinal, 1, image_grade_inv_warp, 1, 0)


        cv2.imshow("Final Image", imageFinal)
    except:
        imageFinal = np.zeros((width, height,3), np.uint8)
        cv2.imshow("Final Image", imageFinal)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Resources/Scanned"+str(count)+".jpg", imageFinal)
        cv2.rectangle(imageFinal, (0, imageFinal.shape[0]//4), (imageFinal.shape[1], int(imageFinal.shape[0]//2.5)), (0,255,0), cv2.FILLED)
        cv2.putText(imageFinal, "Scan Saved", (imageFinal.shape[1]//4, imageFinal.shape[0]//3),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0,0), 2)
        cv2.imshow("Output Image", imageFinal)
        cv2.waitKey(300)
        count+=1
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()