from PIL import Image 
import numpy
from copy import copy, deepcopy
import sys

BW_THRESHOLD = int(sys.argv[1])
BW_LIMIT = int(sys.argv[2])
MB_THRESHOLD = int(sys.argv[3])
FILE_NAME = sys.argv[4]

points_from_line = []

def save_image( npdata, filename ) :
    img = Image.fromarray( npdata.astype('uint8')*255, "L" )
    img.save( filename )

def closestOneToLeft(row, position):
    counter = 0
    position = position - 1
    while position >= 0 and row[position] == 1:
        counter = counter + 1
        position = position - 1
    return counter

def closestOneToRight(row, position):
    counter = 0
    position = position + 1
    while position < len(row) and row[position] == 1:
        counter = counter + 1
        position = position + 1
    return counter

def checkIfShouldBeBlackWithPosition(row, position, GAMMA):
    left = closestOneToLeft(row, position)
    right = closestOneToRight(row, position)
    if left + right < GAMMA:
        return True
    else:
        return False

def doVerticalEditingAndReturnNewMatrix(original_matrix, GAMMA):
    t_matrix = original_matrix.T
    matrix = doHorizontalEditingAndReturnNewMatrix(t_matrix, GAMMA)
    matrix = matrix.T
    return matrix

def doHorizontalEditingAndReturnNewMatrix(original_matrix, GAMMA):
    matrix = deepcopy(original_matrix)
    for row in matrix:
        row_counter = 0
        for pixel in row:
            if pixel == 1:
                checked = checkIfShouldBeBlackWithPosition(row, row_counter, GAMMA)
                if checked:
                    row[row_counter] = 0
            row_counter = row_counter + 1
    return matrix

def andThem(m1, m2, width, height):
    x = 0
    matrix = deepcopy(m1)
    while x < height:
        y = 0
        while y < width:
            if m1[x][y] == 0 and m2[x][y] == 0:
                matrix[x][y] = 0
            else:
                matrix[x][y] = 1
            y = y + 1
        x = x + 1
    return matrix

def checkBlackPixel(pixelArray):
    if pixelArray[0] > BW_THRESHOLD:
        return True
    else:
        return False

def openImageAsArray():
    im= Image.open(FILE_NAME) 
    matrix = numpy.asarray( im, dtype="int32" )
    (width, height) = im.size
    return (width, height, matrix)

def convertToBW(matrix):
    bpmatrix = []
    for z in matrix:
        row = []
        value = 0
        for y in z:
            if (checkBlackPixel(y)):
                value = 1
            else:
                value = 0
            row += [value]
        bpmatrix += [row]
    new_matrix = numpy.array(bpmatrix)
    return new_matrix

def mostlyBlackRow(row):
    black_count = 0
    black_count_threshold = MB_THRESHOLD
    currently_black = False
    for x in row:
        if x == 0:
            if not currently_black:
                currently_black = True
                black_count = black_count + 1
        else:
            if currently_black:
                currently_black = False
    return (black_count > black_count_threshold)

def fillAndDeleteRecursively(original_matrix, top_threshold, bottom_threshold, x, y):
    width = len(original_matrix[0])
    white = 1
    black = 0
    original_matrix[y][x] = white
    if x > 0:
        if original_matrix[y][x-1] == black:
            fillAndDeleteRecursively(original_matrix, top_threshold, bottom_threshold, x-1, y);
    if y > top_threshold:
        if original_matrix[y-1][x] == black:
            fillAndDeleteRecursively(original_matrix, top_threshold, bottom_threshold, x, y-1);
    if x < width-1:
        if original_matrix[y][x+1] == black:
            fillAndDeleteRecursively(original_matrix, top_threshold, bottom_threshold, x+1, y);
    if y < bottom_threshold:
        if original_matrix[y+1][x] == black:
            fillAndDeleteRecursively(original_matrix, top_threshold, bottom_threshold, x, y+1);

def fillAndDelete(original_matrix, middle_threshold, top_threshold, bottom_threshold, column_position):
    current_x = column_position
    current_y = middle_threshold
    fillAndDeleteRecursively(original_matrix, top_threshold, bottom_threshold, current_x, current_y)

def maybeIncreaseBlackCount(matrix, width, height, x, y):
    black = 0
    if x >= 0 and y >= 0 and x < width - 1 and y < height - 1:
        if matrix[y][x] == 0:
            black = 1
    return black    

def checkBlackPosition(matrix, width, height, x, y):
    if x >= 0 and y >= 0 and x < width - 1 and y < height - 1:
        if matrix[y][x] == 0:
            return True
        else:
            return False  
    else:
        return False

def blackPixelBoxThresholdNotExceeded(matrix, x, y, threshold_to_not_exceed, box_size):
    width = len(matrix[0])
    height = len(matrix)
    
    # first count the immediate box
    x_counter = -1
    black_counter = 0
    while(x_counter < 2):
        y_counter = -1
        while(y_counter < 2):
            if x_counter >= 0 and y_counter >= 0 and x_counter < width - 1 and y_counter < height - 1:
                if matrix[y_counter][x_counter] == 0:
                    black_counter = black_counter+1  
            y_counter = y_counter+1 
        x_counter = x_counter+1
    
    # second count the surrounding box
    if checkBlackPosition(matrix, width, height, x-1, y-1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x-2, y-2)
    if checkBlackPosition(matrix, width, height, x-1, y-1) or checkBlackPosition(matrix, width, height, x-1, y):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x-2, y-1)
    if checkBlackPosition(matrix, width, height, x-1, y-1) or checkBlackPosition(matrix, width, height, x-1, y) or checkBlackPosition(matrix, width, height, x-1, y+1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x-2, y)
    if checkBlackPosition(matrix, width, height, x-1, y) or checkBlackPosition(matrix, width, height, x-1, y+1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x-2, y+1)
    if checkBlackPosition(matrix, width, height, x-1, y+1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x-2, y+2)

    if checkBlackPosition(matrix, width, height, x-1, y-1) or checkBlackPosition(matrix, width, height, x, y-1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x-1, y-2)
    if checkBlackPosition(matrix, width, height, x-1, y-1) or checkBlackPosition(matrix, width, height, x, y-1) or checkBlackPosition(matrix, width, height, x+1, y-1):    
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x, y-2)
    if checkBlackPosition(matrix, width, height, x, y-1) or checkBlackPosition(matrix, width, height, x+1, y-1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x+1, y-2)

    if checkBlackPosition(matrix, width, height, x-1, y+1) or checkBlackPosition(matrix, width, height, x, y+1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x-1, y+2)
    if checkBlackPosition(matrix, width, height, x-1, y+1) or checkBlackPosition(matrix, width, height, x, y+1) or checkBlackPosition(matrix, width, height, x+1, y+1):    
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x, y+2)
    if checkBlackPosition(matrix, width, height, x, y+1) or checkBlackPosition(matrix, width, height, x+1, y+1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x+1, y+2)

    if checkBlackPosition(matrix, width, height, x+1, y-1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x+2, y-2)
    if checkBlackPosition(matrix, width, height, x+1, y-1) or checkBlackPosition(matrix, width, height, x+1, y):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x+2, y-1)
    if checkBlackPosition(matrix, width, height, x+1, y-1) or checkBlackPosition(matrix, width, height, x+1, y) or checkBlackPosition(matrix, width, height, x+1, y+1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x+2, y)
    if checkBlackPosition(matrix, width, height, x+1, y) or checkBlackPosition(matrix, width, height, x+1, y+1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x+2, y+1)
    if checkBlackPosition(matrix, width, height, x+1, y+1):
        black_counter = black_counter + maybeIncreaseBlackCount(matrix, width, height, x+2, y+2)

    return (black_counter < threshold_to_not_exceed)


def fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x, y, original_matrix_unchanged):
    height = len(original_matrix)
    width = len(original_matrix[0])
    white = 1
    black = 0
    threshold_to_not_exceed = BW_LIMIT
    box_size = 6
    original_matrix[y][x] = white
    #print(" deleting %0d:%d" % (y, x))
    if x > 0:
        if original_matrix[y][x-1] == black and blackPixelBoxThresholdNotExceeded(original_matrix_unchanged, x-1, y, threshold_to_not_exceed, box_size):
            fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x-1, y, original_matrix_unchanged);
    if y > top_threshold - (bottom_threshold-top_threshold):
        if original_matrix[y-1][x] == black and blackPixelBoxThresholdNotExceeded(original_matrix_unchanged, x, y-1, threshold_to_not_exceed, box_size):
            fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x, y-1, original_matrix_unchanged);
    if x < width-1:
        if original_matrix[y][x+1] == black and blackPixelBoxThresholdNotExceeded(original_matrix_unchanged, x+1, y, threshold_to_not_exceed, box_size):
            fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x+1, y, original_matrix_unchanged);
    if y < bottom_threshold + (bottom_threshold-top_threshold) and y < height-1:
        if original_matrix[y+1][x] == black and blackPixelBoxThresholdNotExceeded(original_matrix_unchanged, x, y+1, threshold_to_not_exceed, box_size):
            fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x, y+1, original_matrix_unchanged);

def fillAndDelete2(original_matrix, row_position, top_threshold, bottom_threshold, column_position):
    matrix_copy = deepcopy(original_matrix)
    current_x = column_position
    current_y = row_position
    fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, current_x, current_y, matrix_copy)

def fillAndfind(original_matrix, top_threshold, bottom_threshold, x, y, original_matrix_unchanged):
    height = len(original_matrix)
    width = len(original_matrix[0])
    white = 1
    black = 0
    toppestX = x
    toppestY = y
    bottomestX = x
    bottomestY = y
    original_matrix[y][x] = white
    if x > 0:
        if original_matrix[y][x-1] == black:
            (bottomX, bottomY, topX, topY) = fillAndfind(original_matrix, top_threshold, bottom_threshold, x-1, y, original_matrix_unchanged);
            if bottomY > bottomestY:
                bottomestX = bottomX
                bottomestY = bottomY
            if topY < toppestX:
                toppestX = topX
                toppestY = topY
    if y > top_threshold:
        if original_matrix[y-1][x] == black:
            (bottomX, bottomY, topX, topY) = fillAndfind(original_matrix, top_threshold, bottom_threshold, x, y-1, original_matrix_unchanged);
            if bottomY > bottomestY:
                bottomestX = bottomX
                bottomestY = bottomY
            if topY < toppestX:
                toppestX = topX
                toppestY = topY
    if x < width-1:
        if original_matrix[y][x+1] == black:
            (bottomX, bottomY, topX, topY) = fillAndfind(original_matrix, top_threshold, bottom_threshold, x+1, y, original_matrix_unchanged);
            if bottomY > bottomestY:
                bottomestX = bottomX
                bottomestY = bottomY
            if topY < toppestX:
                toppestX = topX
                toppestY = topY
    if y < bottom_threshold:
        if original_matrix[y+1][x] == black:
            (bottomX, bottomY, topX, topY) = fillAndfind(original_matrix, top_threshold, bottom_threshold, x, y+1, original_matrix_unchanged);
            if bottomY > bottomestY:
                bottomestX = bottomX
                bottomestY = bottomY
            if topY < toppestX:
                toppestX = topX
                toppestY = topY
    return (bottomestX, bottomestY, toppestX, toppestY)

def calculateBox(original_matrix, row_position, top_threshold, bottom_threshold, column_position):
    matrix_copy = deepcopy(original_matrix)
    current_x = column_position
    current_y = row_position
    matrix_copy_fillAndFind = deepcopy(original_matrix)
    (bottomX, bottomY, topX, topY) = fillAndfind(matrix_copy_fillAndFind, top_threshold, bottom_threshold, current_x, current_y, matrix_copy)
    height = len(original_matrix)
    width = len(original_matrix[0])

    boxWidth = max(abs(bottomX-topX)*2, 80) # THIS SHOULD BE DYNAMIC
    boxHeight = (bottom_threshold-top_threshold)*2

    # Calculate Box One
    firstBoxLeft = topX-boxWidth/2
    firstBoxTop = topY-boxHeight
    firstBoxRight = topX+boxWidth/2
    firstBoxBottom = topY

    # Calculate Box Two 
    secondBoxLeft = bottomX-boxWidth/2
    secondBoxTop = bottomY
    secondBoxRight = bottomX+boxWidth/2
    secondBoxBottom = bottomY+boxHeight

    if secondBoxLeft < 0:
        secondBoxLeft = 0
    if firstBoxLeft < 0:
        firstBoxLeft = 0
    if firstBoxTop < 0:
        firstBoxTop = 0
    if secondBoxTop < 0:
        secondBoxTop = 0

    if firstBoxRight >= width:
        firstBoxRight = width-1
    if secondBoxRight >= width:
        secondBoxRight = width-1
    if firstBoxBottom >= height:
        firstBoxBottom = height-1
    if secondBoxBottom >= height:
        secondBoxBottom = height-1


    #print(" boxes %d:%d:%d:%d:%d:%d:%d:%d" %  (firstBoxLeft, firstBoxTop, firstBoxRight, firstBoxBottom, secondBoxLeft, secondBoxTop, secondBoxRight, secondBoxBottom))
    return (firstBoxLeft, firstBoxTop, firstBoxRight, firstBoxBottom, secondBoxLeft, secondBoxTop, secondBoxRight, secondBoxBottom)

def twoOrLessConnectedAndStop(matrix, x, y):
    connectedCount = 0
    height = len(matrix)
    width = len(matrix[0])
    if x-1 >= 0:
        if matrix[y][x-1] == 0:
            connectedCount = connectedCount + 1 
    if y-1 >= 0:
        if matrix[y-1][x] == 0:
            connectedCount = connectedCount + 1 
    if x+1 <= width -1:
        if matrix[y][x+1] == 0:
            connectedCount = connectedCount + 1 
    if y+1 <= height - 1:
        if matrix[y+1][x] == 0:
            connectedCount = connectedCount + 1
    return (connectedCount <= 2)

def twoOrLessConnected(matrix, x, y):
    connectedCount = 0
    height = len(matrix)
    width = len(matrix[0])
    neightborsAreTwoOrLessConnected = True
    if x-1 >= 0:
        if matrix[y][x-1] == 0:
            neightborsAreTwoOrLessConnected = neightborsAreTwoOrLessConnected and twoOrLessConnectedAndStop(matrix, x-1, y)
    if y-1 >= 0:
        if matrix[y-1][x] == 0:
            neightborsAreTwoOrLessConnected = neightborsAreTwoOrLessConnected and twoOrLessConnectedAndStop(matrix, x, y-1) 
    if x+1 <= width -1:
        if matrix[y][x+1] == 0:
            neightborsAreTwoOrLessConnected = neightborsAreTwoOrLessConnected and twoOrLessConnectedAndStop(matrix, x+1, y)
    if y+1 <= height - 1:
        if matrix[y+1][x] == 0:
            neightborsAreTwoOrLessConnected = neightborsAreTwoOrLessConnected and twoOrLessConnectedAndStop(matrix, x, y+1)
    return twoOrLessConnectedAndStop(matrix, x, y) and neightborsAreTwoOrLessConnected


def removePotentialLinePixelsInBox(original_matrix, boxLeft, boxTop, boxRight, boxBottom, original_unchanged_matrix):
    column_index = boxLeft
    while column_index < boxRight:
        row_index = boxTop
        while row_index < boxBottom:
            if twoOrLessConnected(original_unchanged_matrix, column_index, row_index):
                original_matrix[row_index][column_index] = 1 # black for now
            row_index = row_index + 1
        column_index = column_index+1


def startRemovingBasedOnMiddle(top_threshold, bottom_threshold, middle_threshold, original_matrix):
    matrix_copy = deepcopy(original_matrix) 

    column_position = 0
    print(middle_threshold)
    while column_position < len(original_matrix[middle_threshold]) - 1:
        if original_matrix[middle_threshold][column_position] == 0:
            fillAndDelete2(original_matrix, middle_threshold, top_threshold, bottom_threshold, column_position)
            #save_image(original_matrix, "1.jpeg")
        column_position = column_position + 1

    column_position = 0
    while column_position < len(original_matrix[middle_threshold]) - 1:
        row_position = top_threshold
        while row_position < bottom_threshold:
            if matrix_copy[row_position][column_position] == 0:
                (firstBoxLeft, firstBoxTop, firstBoxRight, firstBoxBottom, secondBoxLeft, secondBoxTop, secondBoxRight, secondBoxBottom) = calculateBox(matrix_copy, row_position, top_threshold, bottom_threshold, column_position)
                removePotentialLinePixelsInBox(original_matrix, firstBoxLeft, firstBoxTop, firstBoxRight, firstBoxBottom, matrix_copy)
                removePotentialLinePixelsInBox(original_matrix, secondBoxLeft, secondBoxTop, secondBoxRight, secondBoxBottom, matrix_copy)
                #save_image(original_matrix, "2.jpeg")
            row_position = row_position + 1
        column_position = column_position + 1

def doThingWithTopAndBottom(matrix, top_threshold, bottom_threshold, original_matrix):
    middle_threshold = top_threshold + (bottom_threshold-top_threshold)/2
    print(middle_threshold)
    startRemovingBasedOnMiddle(top_threshold, bottom_threshold, middle_threshold, original_matrix)

def run(matrix, original_matrix):
    top_threshold = 0
    bottom_threshold = 0
    lookingForTopThreshold = False
    d_total_count = 0
    for row in matrix:
        if lookingForTopThreshold:
            if (not mostlyBlackRow(matrix[top_threshold])):
                lookingForTopThreshold = False
                top_threshold = top_threshold + 1
                bottom_threshold = top_threshold
            else:
                top_threshold = top_threshold + 1
        else:
            if mostlyBlackRow(matrix[bottom_threshold]) or bottom_threshold == len(matrix)-1:
                doThingWithTopAndBottom(matrix, top_threshold, bottom_threshold, original_matrix)
                lookingForTopThreshold = True
                bottom_threshold = bottom_threshold + 1
                top_threshold = bottom_threshold
            else:
                bottom_threshold = bottom_threshold + 1
        d_total_count = d_total_count + 1

def runHoritontal(matrix, original_matrix):
    top_threshold = 0
    bottom_threshold = 0
    lookingForTopThreshold = False
    d_total_count = 0
    for row in matrix:
        if lookingForTopThreshold:
            if (not mostlyBlackRow(matrix[top_threshold])):
                lookingForTopThreshold = False
                top_threshold = top_threshold + 1
                bottom_threshold = top_threshold
            else:
                top_threshold = top_threshold + 1
        else:
            if mostlyBlackRow(matrix[bottom_threshold]) or bottom_threshold == len(matrix)-1:
                doThingWithTopAndBottom(matrix, top_threshold, bottom_threshold, original_matrix)
                lookingForTopThreshold = True
                bottom_threshold = bottom_threshold + 1
                top_threshold = bottom_threshold
            else:
                bottom_threshold = bottom_threshold + 1
        d_total_count = d_total_count + 1

def left_edge_connected(matrix, y, x):
    if matrix[y-1][x-1] == 0 or matrix[y][x-1] == 0 or matrix[y+1][x-1] == 0:
        return True
    else:
        return False

def right_edge_connected(matrix, y, x):
    if matrix[y-1][x+1] == 0 or matrix[y][x+1] == 0 or matrix[y+1][x+1] == 0:
        return True
    else:
        return False

def top_edge_connected(matrix, y, x):
    if matrix[y-1][x-1] == 0 or matrix[y][x] == 0 or matrix[y+1][x+1] == 0:
        return True
    else:
        return False

def bottom_edge_connected(matrix, y, x):
    if matrix[y+1][x-1] == 0 or matrix[y+1][x] == 0 or matrix[y+1][x+1] == 0:
        return True
    else:
        return False

def makeBoxCircumferenceForPixel(matrix, y, x, boxSize):
    boxCircumference = []
    leftBoundary = x - boxSize/2
    topBoundary = y - boxSize/2
    rightBoundary = x + boxSize/2
    bottomBoundary = y + boxSize/2
    width = len(matrix[0])
    height = len(matrix)

    x = leftBoundary
    while x != rightBoundary:
        boxCircumference = boxCircumference + [matrix[topBoundary][x]]
        x = x + 1

    y = topBoundary+1
    while y != bottomBoundary:
        boxCircumference = boxCircumference + [matrix[y][rightBoundary]]
        y = y + 1

    x = rightBoundary-1
    while x != leftBoundary:
        boxCircumference = boxCircumference + [matrix[bottomBoundary][x]]
        x = x - 1

    y = bottomBoundary-1
    while y != topBoundary+1:
        boxCircumference = boxCircumference + [matrix[y][leftBoundary]]
        y = y - 1

    return boxCircumference

def boxTotal(matrix, y, x, boxSize):
    column_index = x - boxSize/2
    row_index = y - boxSize/2
    columnMax = x + boxSize/2
    rowMax = y + boxSize/2
    blackCount = 0
    while row_index < rowMax:
        column_index = x - boxSize/2
        while column_index < columnMax:
            if matrix[row_index][column_index] == 0:
                blackCount = blackCount + 1
            column_index = column_index + 1
        row_index = row_index + 1
    return blackCount

def twoContinuousOfLess2SegmentsInBox(boxCirumference, boxSize):
    countToPreventInfinateLoop = len(boxCirumference)
    while boxCirumference[0] == 0 and countToPreventInfinateLoop > 0:
        countToPreventInfinateLoop = countToPreventInfinateLoop-1
        lastIndex = len(boxCirumference)-1
        temp = boxCirumference[lastIndex]
        del boxCirumference[lastIndex]
        boxCirumference = [temp] + boxCirumference
    #print(boxCirumference)
    currentContCount = 0
    currentSegmentCount = 0
    currentlyBlack = False
    boxLength = len(boxCirumference)
    #print "boxlen - %d" % (boxLength)
    lengthThreshold = boxSize
    lengthCounter = 0
    boxIndex = 0
    firstSegmentFound = False
    while boxIndex < boxLength - 1:
        #print "box index = %d" % (boxIndex)
        if not currentlyBlack:
            #print "not black"
            if boxCirumference[boxIndex] == 0:
                #print "found first black"
                currentlyBlack = True
                currentSegmentCount = currentSegmentCount + 1
                if currentSegmentCount > 2:
                    #print "false cause more segments"
                    return False
                currentContCount = currentContCount + 1
                if lengthCounter < lengthThreshold and firstSegmentFound:
                    #print "FALSE"
                    return False
            else:
                if firstSegmentFound:
                    #print "increase it"
                    lengthCounter = lengthCounter + 1
        else:
            if boxCirumference[boxIndex] == 0:
                #print "conintue black"
                currentContCount = currentContCount + 1
                if currentContCount > 2:
                    #print "false for contunity count"
                    return False
            else:
                #print "now not black"
                currentlyBlack = False
                currentContCount = 0
                firstSegmentFound = True
                lengthCounter = 1
        boxIndex = boxIndex + 1
    if currentSegmentCount == 2:
        return True
    else:
        return False

def connectedInBoxTotalRecursively(matrix, y, x, leftBoundary, topBoundary, rightBoundary, bottomBoundary):
    matrix[y][x] = 1
    count = 1
    if x-1 >= leftBoundary and matrix[y][x-1] == 0:
        count = count + connectedInBoxTotalRecursively(matrix, y, x-1, leftBoundary, topBoundary, rightBoundary, bottomBoundary)
    if x+1 <= rightBoundary and matrix[y][x+1] == 0:
        count = count + connectedInBoxTotalRecursively(matrix, y, x+1, leftBoundary, topBoundary, rightBoundary, bottomBoundary)
    if y-1 >= topBoundary and matrix[y-1][x] == 0:
        count = count + connectedInBoxTotalRecursively(matrix, y-1, x, leftBoundary, topBoundary, rightBoundary, bottomBoundary)
    if y+1 >= bottomBoundary and matrix[y+1][x] == 0:
        count = count + connectedInBoxTotalRecursively(matrix, y+1, x, leftBoundary, topBoundary, rightBoundary, bottomBoundary)
    return count

def connectedInBoxTotal(matrix, y, x, boxSize):
    matrix_copy = deepcopy(matrix)
    leftBoundary = x - boxSize/2
    topBoundary = y - boxSize/2
    rightBoundary = x + boxSize/2
    bottomBoundary = y + boxSize/2
    connectedTotal = connectedInBoxTotalRecursively(matrix, y, x, leftBoundary, topBoundary, rightBoundary, bottomBoundary)
    return connectedTotal


def check_line_candidate(matrix, y, x):
    boxSize = 6
    boxThreshold = 11
    boxCirumference = makeBoxCircumferenceForPixel(matrix, y, x, boxSize)
    theBoxTotal = boxTotal(matrix, y, x, boxSize)
    #print(boxCirumference)
    #print(theBoxTotal)
    if theBoxTotal < boxThreshold and theBoxTotal > boxSize:
        #print "here"
        if twoContinuousOfLess2SegmentsInBox(boxCirumference, boxSize):
            bt = connectedInBoxTotal(matrix, y, x, boxSize)
            #print "oh...... potential with total %d:%d" % (theBoxTotal, bt)
            #if connectedInBoxTotal(matrix, y, x, boxSize) == theBoxTotal:
            return True
    return False

def removeLine(matrix, y, x):
    height = len(matrix)
    width = len(matrix[0])
    white = 1
    black = 0
    threshold_to_not_exceed = 5
    box_size = 6
    matrix[y][x] = white
    global points_from_line
    points_from_line = points_from_line + [(y,x)]
    #print(" deleting %0d:%d" % (y, x))
    if x > 0:
        if matrix[y][x-1] == black and blackPixelBoxThresholdNotExceeded(matrix, x-1, y, threshold_to_not_exceed, box_size):
            removeLine(matrix, y, x-1);
    if y > 0:
        if matrix[y-1][x] == black and blackPixelBoxThresholdNotExceeded(matrix, x, y-1, threshold_to_not_exceed, box_size):
            removeLine(matrix, y-1, x);
    if x < width-1:
        if matrix[y][x+1] == black and blackPixelBoxThresholdNotExceeded(matrix, x+1, y, threshold_to_not_exceed, box_size):
            removeLine(matrix, y, x+1);
    if y < height-1:
        if matrix[y+1][x] == black and blackPixelBoxThresholdNotExceeded(matrix, x, y+1, threshold_to_not_exceed, box_size):
            removeLine(matrix, y+1, x);

def removeLines(matrix):
    matrix_copy = deepcopy(matrix) 
    height = len(matrix_copy)
    width = len(matrix_copy[0])
    column_index = 3
    while column_index < (height-4):
        row_index = 3
        while row_index < (width-4):
            if matrix_copy[column_index][row_index] == 0:
                if check_line_candidate(matrix_copy, column_index, row_index):
                    #print(" deleting %0d:%d" % (column_index, row_index))
                    removeLine(matrix_copy, column_index, row_index)
            row_index = row_index+1
        column_index = column_index + 1
    return matrix_copy

def getBlackPixelCount(matrix):
    height = len(matrix)
    width = len(matrix[0])
    column_index = 0
    black_pixel_counter = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if matrix[column_index][row_index] == 0:
                black_pixel_counter = black_pixel_counter + 1
            row_index = row_index+1
        column_index = column_index + 1
    return black_pixel_counter

def getMinXandXlength(matrix):
    height = len(matrix)
    width = len(matrix[0])
    column_index = 0
    firstFound = False
    minBlack = 0
    maxBlack = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if matrix[column_index][row_index] == 0:
                if not firstFound:
                    firstFound = True
                    minBlack = row_index
                    maxBlack = row_index
                else:
                    if row_index < minBlack:
                        minBlack = row_index
                    if row_index > maxBlack:
                        maxBlack = row_index
            row_index = row_index+1
        column_index = column_index + 1
    return (minBlack, maxBlack)    

def getMinYandYlength(matrix):
    height = len(matrix)
    width = len(matrix[0])
    column_index = 0
    firstFound = False
    minBlack = 0
    maxBlack = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if matrix[column_index][row_index] == 0:
                if not firstFound:
                    firstFound = True
                    minBlack = column_index
                    maxBlack = column_index
                else:
                    if column_index < minBlack:
                        minBlack = column_index
                    if column_index > maxBlack:
                        maxBlack = column_index
            row_index = row_index+1
        column_index = column_index + 1
    return (minBlack, maxBlack)  

def getNumberOriginalBlackPixels(new_matrix, original_matrix):
    height = len(new_matrix)
    width = len(new_matrix[0])
    column_index = 0
    blackCount = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if new_matrix[column_index][row_index] == 0:
                if original_matrix[column_index][row_index] == 0:
                    blackCount = blackCount + 1
            row_index = row_index+1
        column_index = column_index + 1
    return  blackCount

def getNumberOfTransitions(matrix, original_matrix):
    height = len(matrix)
    width = len(matrix[0])
    column_index = 0
    transitionCount = 0
    
    while column_index < (height-1):
        row_index = 0
        current_row_transition_count = 0
        currentlyBlack = False
        
        firstFound = False
        while row_index < (width-1):
            
            if matrix[column_index][row_index] == 0:

                if not firstFound:
                    firstFound = True
                    if original_matrix[column_index][row_index] == 0:
                        currentlyBlack = True
                    else:
                        currentlyBlack = False

                if original_matrix[column_index][row_index] == 0:
                    if not currentlyBlack:
                        transitionCount = transitionCount + 1
                        currentlyBlack = True
                    else:
                        if currentlyBlack:
                            currentlyBlack = False

            row_index = row_index+1
        column_index = column_index + 1
        transitionCount = transitionCount + current_row_transition_count
    return  transitionCount    

def getBWmatrix():
    (width, height, matrix) = openImageAsArray()
    bwMatrix = convertToBW(matrix)
    return (width, height, bwMatrix) 

def fillMatrix(matrix, y, x):
    height = len(matrix)
    width = len(matrix[0])
    #print("x: %d y: %d" % (x, y))
    #print("fill")
    #print("%d %d %d %d" % (matrix[y][x-1], matrix[y][x+1], matrix[y-1][x], matrix[y+1][x]))
    matrix[y][x] = 2

    foundOne = False
    while(True):
        # find a two, check all around it
        column_index = 0
        while column_index < (height-1):
            row_index = 0
            while row_index < (width-1):
                if matrix[column_index][row_index] == 2:
                    #print("Found a two %d %d" % (column_index, row_index))
                    # Check all around it and make it 2's
                    if matrix[column_index][row_index-1] == 0:
                        matrix[column_index][row_index-1] = 2
                        foundOne = True
                    if matrix[column_index][row_index+1] == 0:
                        matrix[column_index][row_index+1] = 2
                        foundOne = True
                    if matrix[column_index-1][row_index] == 0:
                        matrix[column_index-1][row_index] = 2
                        foundOne = True
                    if matrix[column_index+1][row_index] == 0:
                        matrix[column_index+1][row_index] = 2
                        foundOne = True
                row_index = row_index+1
            column_index = column_index + 1
        if row_index == width-1 and column_index == height-1:
            return 

def set_to_zero(matrix):
    height = len(matrix)
    width = len(matrix[0])
    column_index = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            matrix[column_index][row_index] = 1
            row_index = row_index+1
        column_index = column_index + 1

def changeAllButTwo(matrix):
    height = len(matrix)
    width = len(matrix[0])
    column_index = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if not (matrix[column_index][row_index] == 2):
                matrix[column_index][row_index] = 1
            if (matrix[column_index][row_index] == 2):
                matrix[column_index][row_index] = 0
            row_index = row_index+1
        column_index = column_index + 1

def checkIfShouldBeDeleted(matrix, original_matrix):
    delta_x = getCalculations(matrix, original_matrix)
    if delta_x < 200:
        return True
    else:
        return False

def getCalculations(matrix, original_matrix):
    T = getNumberOfTransitions(matrix, original_matrix)
    BC = getBlackPixelCount(matrix)
    DC = getNumberOriginalBlackPixels(matrix, original_matrix)
    (mi_y, ma_y) = getMinYandYlength(matrix)
    delta_y = ma_y-mi_y
    (mi_x, ma_x) = getMinXandXlength(matrix)
    delta_x = ma_x-mi_x
    
    e  = 0
    s = 0
    Rm = 0
    if not (delta_y == 0):
        e = delta_x/delta_y
    else:
        e = 0

    if not (e == 0):
        s = int(BC/e)
    else:
        s = 0

    if not (DC == 0):
        Rm = int(BC/DC)
    else:
        Rm = 0

    print("###############################################")
    print("T = %d" % (T))
    print("BC = %d" % (BC))
    print("DC = %d" % (DC))
    print("y = %d"  % (delta_y))
    print("x = %d"  % (delta_x))
    print("s = %d"  % (s))
    print("e = %d"  % (e))
    print("Rm = %d"  % (Rm))
    return delta_x

def drawOnPiece(canvas, matrix_copy):
    height = len(matrix_copy)
    width = len(matrix_copy[0])
    column_index = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if matrix_copy[column_index][row_index] == 0:
                canvas[column_index][row_index] = 0
            row_index = row_index + 1
        column_index = column_index + 1

def deleteMatrixFromMatrix(map_matrix, matrix):
    height = len(matrix)
    width = len(matrix[0])
    column_index = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if map_matrix[column_index][row_index] == 0:
                matrix[column_index][row_index] = 1
            row_index = row_index + 1
        column_index = column_index + 1

def loopThroughConnectedComponents(matrix, original_matrix):
    matrix_of_pieces_to_delete = deepcopy(matrix)
    set_to_zero(matrix_of_pieces_to_delete)
    matrix_copy = deepcopy(matrix) 
    height = len(matrix)
    width = len(matrix[0])
    column_index = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if matrix_copy[column_index][row_index] == 0:
                fillMatrix(matrix_copy, column_index, row_index) # We are just doing one connected component for now
                changeAllButTwo(matrix_copy)
                #save_image(matrix_copy, "after_two.jpeg")
                if checkIfShouldBeDeleted(matrix_copy, original_matrix):
                    drawOnPiece(matrix_of_pieces_to_delete, matrix_copy)
                #
                # Delete this from original matrix
                deleteMatrixFromMatrix(matrix_copy, matrix)
                matrix_copy = deepcopy(matrix)
                #
                #return matrix_of_pieces_to_keep
            row_index = row_index+1
        column_index = column_index + 1
    return matrix_of_pieces_to_delete

def drawPiecesOntoCanvas(canvas_matrix, map_matrix, original_matrix):
    height = len(original_matrix)
    width = len(original_matrix[0])
    column_index = 0
    while column_index < (height-1):
        row_index = 0
        while row_index < (width-1):
            if map_matrix[column_index][row_index] == 0:
                if original_matrix[column_index][row_index] == 0:
                    canvas_matrix[column_index][row_index] = 0
                else:
                    canvas_matrix[column_index][row_index] = 1
            else:
                canvas_matrix[column_index][row_index] = 1
            row_index = row_index+1
        column_index = column_index + 1

def drawLineBackOne(matrix):
    for (y,x) in points_from_line:
        matrix[y][x] = 0

####################################################
# MAIN
####################################################
(width, height, bwMatrix) = getBWmatrix()

# Step One: Remove Line and Make Copy
lineRemovedMatrix = removeLines(bwMatrix)
save_image(lineRemovedMatrix, "after_lines_removed.jpeg")

# Step Two: Do RLSA
mod_matrix = doHorizontalEditingAndReturnNewMatrix(lineRemovedMatrix, 300)
save_image(mod_matrix, "after_hor.jpeg")
mod_matrix2 = doVerticalEditingAndReturnNewMatrix(lineRemovedMatrix, 500)
save_image(mod_matrix2, "after_ver.jpeg")
final_matrix = andThem(mod_matrix, mod_matrix2, width, height)
save_image(final_matrix, "final.jpeg")

# Step Three: Loop through connected components
print("here")
matrix_of_pieces_to_delete = loopThroughConnectedComponents(final_matrix, lineRemovedMatrix)
save_image(matrix_of_pieces_to_delete, "matrix_of_pieces_to_delete.jpeg")

#drawLineBackOne(matrix_to_remove_line)
#save_image(matrix_to_remove_line, "matrix_to_remove_line1.jpeg")
deleteMatrixFromMatrix(matrix_of_pieces_to_delete, bwMatrix)
save_image(bwMatrix, "matrix_to_remove_line2.jpeg")

# Step Four: Finally run remove line
lineRemovedMatrix = removeLines(bwMatrix)
save_image(lineRemovedMatrix, "1.jpeg")
run(bwMatrix, bwMatrix)
save_image(bwMatrix, "complete.jpeg")
lineRemovedMatrix = removeLines(bwMatrix)
save_image(lineRemovedMatrix, "2.jpeg")





