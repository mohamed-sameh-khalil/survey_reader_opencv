import math
import numpy as np
import cv2
from CONST import ans, analyzer

# given an angle and an image, this function will return a the image rotated by this angle
def rotateImage(image, angle): # source: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    # rotation is clockwise
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


#return 2 images one with only circles and one with only squares
def seperateSquaresAndCircles(img):
    #this mask are for cleaning noise around circle
    cmask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
    # threshhold image and then open to keep only circles
    circles = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)[1]
    circles = cv2.erode(circles,cmask)
    circles = cv2.dilate(circles,cmask)
    #those masks are for cleaning noise around squares 
    mask = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
    mask2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    
    squares = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)[1] - circles
    squares = cv2.erode(squares,mask)
    squares = cv2.dilate(squares,mask2)
    return circles,squares

# returns 2 lists one with the top 3 horizontal squares and one with 
# the multiple vertical squares
# separation assumes that top right squares are larger in size
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

# returns a list with the centroids of the circles(choices)
def getCirclesCentroids(circles):
    # I added the opening because some very short white line was missing up my components
    mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
    circles = cv2.erode(circles,mask)
    circles = cv2.dilate(circles,mask)

    cc = cv2.connectedComponentsWithStats(circles)
    centers = []
    for i,c in enumerate(cc[2]):
        # TODO make this part prune to scale
        if (c[cv2.CC_STAT_AREA] > 100 and c[cv2.CC_STAT_AREA] < 10000)\
                and (c[cv2.CC_STAT_LEFT] > 200 and c[cv2.CC_STAT_LEFT] < 1600)\
                and (c[cv2.CC_STAT_TOP] > 150 and c[cv2.CC_STAT_TOP] < 2200): 
            centers.append(tuple(cc[3][i]))
    return centers

# this function calculates rotation angle and decided what is the best rotation angle to fix this rotations
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

# compare with all choices and chooses the closest choice from a list of guesses
def getBestGuess(val, guessbook):
    min = 1000
    best = 0
    for i in guessbook:
        if min > abs(i - val):
            min = abs(i - val)
            best = i
    return guessbook[best]

# given the center of a choice thi function returns a string with question title and its answer
def getAnswer(cc): 
    x = cc[0]
    y = cc[1]
    # guess the question
    i = getBestGuess(y,analyzer)
    # i[0] is the question title, i[1] is list of possible answers
    st = i[0]
    # guess the answer
    i = getBestGuess(x,i[1])
    st += ": " + i
    return st
# index = 1

if __name__ == "__main__":
    for num in range(1, 120):
        # assuming all file names are test_sample#.jpg
        name = "samples/test_sample" + str(num) + ".jpg"
        outname = "samples/test_sample" + str(num) + "_out.jpg"

        # opening the text file
        txtfilename = "samples/test_sample" + str(num) + "_out.txt"
        txtfile = open(txtfilename, "w")

        # if we reach the end of the files break
        img = cv2.imread(name,0)
        if img is None : break

        # get an image with the squares seperated from and another with the small black squares
        circles, squares = seperateSquaresAndCircles(img)
        # seperate the 3 horizontal squares from the rest of the vertical square
        horiz, vert = seperateSquares(squares)
        # from the vertical squares try to guess the correct rotation angle to straighten out the image
        angle = fixRotation(vert)

        # rotate the original image by the correctly guessed image
        img = rotateImage(img,angle)

        # separate circles and squares again and separate horizontal and vertical squares
        circles, squares = seperateSquaresAndCircles(img)
        horiz, vert = seperateSquares(squares)
        # get a list of centroids of each circle(choice)
        cc = getCirclesCentroids(circles)

        # take any the squares as references for unwanted translations in the horizontal and vertical directions
        yref = horiz[0][1]
        xref = vert[0][0]

        # save the relative position of each circle
        locs = []
        for i,c in enumerate(cc):
            pos = c[0] - xref, c[1] - yref
            locs.append(pos)
        
        # for each relative position try to guess the question from Y-component of the circle and the answer from the 
        # X-component of the circle
        result = []
        for c in locs:
            text = getAnswer(c)
            txtfile.write(text + "\n")
            img = cv2.putText(img,text,(25,int(c[1]) + 65),cv2.FONT_HERSHEY_DUPLEX,1,0,thickness=2)
        txtfile.close()
        scale = 3
        img = cv2.resize(img,(img.shape[1] // scale,img.shape[0] // scale))
        cv2.imwrite(outname,img)


