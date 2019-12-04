import math
import numpy as np
import cv2
ans = ["male",
"summer",
"manf",
"neutral",
"strongly agree",
"neutral",
"strongly agree",
"agree",
"disagree",
"disagree",
"disagree",
"agree",
"agree",
"strongly agree",
"neutral",
"strongly agree",
"agree",
"strongly agree",
"disagree",
"strongly agree",
"neutral",
"agree"]

analyzer = {240 : ("Gender", {1200 : "Male", 1270 : "Female"} ),
323 : ("Semester", {966 : "Summer", 433 : "Fall", 700 : "Spring" } ),
404 : ("Major", {867 : "ERGY", 1136 : "MANF", 1000 : "COMM", 733 : "CESS", 600 : "BLDG", 466 : "ENVR", 332 : "MCTA" } ),
444	: ("Major", { 733 : "HAUD", 600 : "CISE", 466 : "MATL", 332 : "LAAR" } ),
913 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
965 : ("Q2", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1003 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1044 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1084 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1084 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1205 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1245 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1284 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1324 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1361 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1403 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1521 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1562 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1602 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1723 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1764 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1842 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
1960 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} ),
2001 : ("Q1", {1000 : "Strongly Agree", 1100: "Agree", 1200 : "Neutral", 1300 : "Disagree", 1400 : "Strongly Disagree"} )}

def rotateImage(image, angle): # source: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    # rotation is clockwise
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def getRotationAngle(point, angle):
    x1 = point[0]
    y1 = point[1]
    x2 = x1 * math.cos(math.radians(angle)) - x1 * math.sin(math.radians(angle))
    if x2 > 500:
        return angle + 180
    return angle

#return 2 images one with only circles and one with only squares
def seperateSquaresAndCircles(img):
    mask = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
    mask2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    circles = cv2.threshold(img,1,255,cv2.THRESH_BINARY_INV)[1]
    squares = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)[1] - circles
    squares = cv2.erode(squares,mask)
    squares = cv2.dilate(squares,mask2)
    return circles,squares
def seperateSquares(squares):
    cc = cv2.connectedComponentsWithStats(squares)
    vert = []
    horiz = []
    for i,c in enumerate(cc[2]):
        if (c[cv2.CC_STAT_AREA] > 200 and c[cv2.CC_STAT_AREA] < 10000):
            if(c[cv2.CC_STAT_AREA] < 400):
                vert.append(cc[3][i])
            else:
                horiz.append(cc[3][i])
    return horiz, vert
def getCirclesCentroids(circles):
    # I added the opening because some very short white line was missing up my components
    mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
    circles = cv2.erode(circles,mask)
    circles = cv2.dilate(circles,mask)

    cc = cv2.connectedComponentsWithStats(circles)
    centers = []
    for i,c in enumerate(cc[2]):
        # TODO make this part prone to scale
        if (c[cv2.CC_STAT_AREA] > 100 and c[cv2.CC_STAT_AREA] < 10000)\
                and (c[cv2.CC_STAT_LEFT] > 200 and c[cv2.CC_STAT_LEFT] < 1600)\
                and (c[cv2.CC_STAT_TOP] > 150 and c[cv2.CC_STAT_TOP] < 2200): 
            centers.append(tuple(cc[3][i]))
    return centers
def markSquares(img, vert, horiz):
    squares = img.copy()
    for i in vert:
        squares = cv2.circle(squares,tuple(map(int,i)),9,255)
    for i in horiz:
        squares = cv2.circle(squares,tuple(map(int,i)),20,255)
    return squares

def fixRotation(vert):
    angle =(math.degrees(math.atan((vert[0][1] - vert[-1][1]) / (vert[0][0] - vert[-1][0]))))# angle with the horiz axis
    if angle < 0 : angle = 270 - angle
    else : angle = 90 - angle
    x1 = vert[0][0]
    y1 = vert[0][1]
    x2 = x1 * math.cos(math.radians(angle)) - x1 * math.sin(math.radians(angle))
    if x2 > 500:
        angle += 180
    return -angle

def getBestGuess(val, guessbook):
    min = 1000
    best = 0
    for i in guessbook:
        if min > abs(i - val):
            min = abs(i - val)
            best = i
    return guessbook[best]

def getAnswer(cc): # this function takes the centers after adjusting according to the reference
    x = cc[0]
    y = cc[1]
    stop = False
    i = getBestGuess(y,analyzer)
    st = i[0]
    i = getBestGuess(x,i[1])
    st += ": " + i
    return st
# index = 1

if __name__ == "__main__":
    for num in range(11, 120):
        name = "samples/test_sample" + str(num) + ".jpg"
        outname = "samples/test_sample" + str(num) + "_out.jpg"
        img = cv2.imread(name,0)
        if img is None : break
        circles, squares = seperateSquaresAndCircles(img) 
        horiz, vert = seperateSquares(squares)
        angle = fixRotation(vert)

        img = rotateImage(img,angle)
        circles, squares = seperateSquaresAndCircles(img)
        # cv2.imwrite("tmp1.jpg",circles)
        # cv2.imwrite("tmp2.jpg",squares)
        # exit()
        cc = getCirclesCentroids(circles)
        horiz, vert = seperateSquares(squares)
        yref = horiz[0][1]
        xref = vert[0][0]
        locs = []
        for i,c in enumerate(cc):
            pos = c[0] - xref, c[1] - yref
            locs.append(pos)
        result = []
        for c in locs:
            text = getAnswer(c)
            img = cv2.putText(img,text,(25,int(c[1]) + 65),cv2.FONT_HERSHEY_DUPLEX,1,0,thickness=2)
        scale = 3
        img = cv2.resize(img,(img.shape[1] // scale,img.shape[0] // scale))
        cv2.imwrite(outname,img)


