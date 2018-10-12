from PIL import Image 
import numpy
from copy import copy, deepcopy

def save_image( npdata, filename ) :
    img = Image.fromarray( npdata.astype('uint8')*255, "L" )
    img.save( filename )

def openImageAndReturnAsMatrixWithWidthAndHeight():
    image_file = Image.open("doc.jpeg") 
    image_file = image_file.convert('1') 
    image_file.save( "out1.jpeg" )
    (width, height) = image_file.size
    matrix = numpy.asarray( image_file, dtype="int32" )
    return (width, height, matrix)

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

##########################################
# MAIN
#########################################
#(width, height, matrix1) = openImageAndReturnAsMatrixWithWidthAndHeight()
#print(matrix1)
#mod_matrix = doHorizontalEditingAndReturnNewMatrix(matrix, 150)
#save_image(mod_matrix, "after_hor.jpeg")
#mod_matrix2 = doVerticalEditingAndReturnNewMatrix(matrix, 100)
#save_image(mod_matrix2, "after_ver.jpeg")
#final_matrix = andThem(mod_matrix, mod_matrix2, width, height)
#save_image(final_matrix, "final.jpeg")
#print mod_matrix

def checkBlackPixel(pixelArray):
    if pixelArray[0] > 190:
        return True
    else:
        return False

def openImageAsArray():
    im= Image.open("docu.jpeg") 
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

#def mostlyBlackRow(row):
#    black_count = 0
#    long_enough = False
#    black_count_threshold = 30
#    for x in row:
#        if x == 0:
#            black_count = black_count + 1
#            if black_count > black_count_threshold:
#                long_enough = True
#        else:
#            black_count = 0
#    return long_enough
def mostlyBlackRow(row):
    black_count = 0
    black_count_threshold = 30
    for x in row:
        if x == 0:
            black_count = black_count + 1
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

def blackPixelBoxThresholdNotExceeded(matrix, x, y, threshold_to_not_exceed, box_size):
    x_counter = x-box_size/2
    black_counter = 0
    while(x_counter< x + box_size):
        x_counter = x_counter+1
        y_counter = y-box_size/2
        while(y_counter < y+ box_size):
            y_counter = y_counter+1
            #print(matrix[y_counter][x_counter])
            if matrix[y_counter][x_counter] == 0:
                black_counter = black_counter+1
    #print "blackness = "
    #print black_counter
    return (black_counter < threshold_to_not_exceed)


def fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x, y, original_matrix_unchanged):
    height = len(original_matrix)
    width = len(original_matrix[0])
    white = 1
    black = 0
    threshold_to_not_exceed = 18 + 3
    box_size = 6
    original_matrix[y][x] = white
    if x > 0:
        if original_matrix[y][x-1] == black and blackPixelBoxThresholdNotExceeded(original_matrix_unchanged, x-1, y, threshold_to_not_exceed, box_size):
            fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x-1, y, original_matrix_unchanged);
    if y > top_threshold - (bottom_threshold-top_threshold):
        if original_matrix[y-1][x] == black and blackPixelBoxThresholdNotExceeded(original_matrix_unchanged, x, y-1, threshold_to_not_exceed, box_size):
            fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x, y-1, original_matrix_unchanged);
    if x < width-1:
        if original_matrix[y][x+1] == black and blackPixelBoxThresholdNotExceeded(original_matrix_unchanged, x+1, y, threshold_to_not_exceed, box_size):
            fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x+1, y, original_matrix_unchanged);
    if y < bottom_threshold + (bottom_threshold-top_threshold):
        if original_matrix[y+1][x] == black and blackPixelBoxThresholdNotExceeded(original_matrix_unchanged, x, y+1, threshold_to_not_exceed, box_size):
            fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, x, y+1, original_matrix_unchanged);

def fillAndDelete2(original_matrix, middle_threshold, top_threshold, bottom_threshold, column_position):
    matrix_copy = deepcopy(original_matrix)
    current_x = column_position
    current_y = middle_threshold
    fillAndDeleteRecursively2(original_matrix, top_threshold, bottom_threshold, current_x, current_y, matrix_copy)

def startRemovingBasedOnMiddle(top_threshold, bottom_threshold, middle_threshold, original_matrix):
    column_position = 0
    while column_position < len(original_matrix[middle_threshold]) - 1:
        if original_matrix[middle_threshold][column_position] == 0:
            fillAndDelete2(original_matrix, middle_threshold, top_threshold, bottom_threshold, column_position)
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
                print "top = "
                print(top_threshold)
                top_threshold = top_threshold + 1
                bottom_threshold = top_threshold
            else:
                top_threshold = top_threshold + 1
                print "topitr = "
                print(top_threshold)
        else:
            if mostlyBlackRow(matrix[bottom_threshold]) or bottom_threshold == len(matrix)-1:
                print "bottom = "
                print(bottom_threshold)
                doThingWithTopAndBottom(matrix, top_threshold, bottom_threshold, original_matrix)
                lookingForTopThreshold = True
                bottom_threshold = bottom_threshold + 1
                top_threshold = bottom_threshold
            else:
                bottom_threshold = bottom_threshold + 1
                print "bottomitr = "
                print(bottom_threshold)
        d_total_count = d_total_count + 1


(width, height, matrix) = openImageAsArray()
print(matrix)
bwMatrix = convertToBW(matrix)
save_image( bwMatrix, "bp.jpeg" )
mod_matrix = doHorizontalEditingAndReturnNewMatrix(bwMatrix, 60)
save_image(mod_matrix, "after_hor.jpeg")

##
run(bwMatrix, bwMatrix)
save_image(bwMatrix, "after_run.jpeg")
##

#mod_matrix2 = doVerticalEditingAndReturnNewMatrix(bwMatrix, 10)
#save_image(mod_matrix2, "after_ver.jpeg")
print(mod_matrix)
#print(mod_matrix2)
#final_matrix = andThem(mod_matrix, mod_matrix2, width, height)
#save_image(final_matrix, "final.jpeg")
print mod_matrix




























    column_position = 0
    print(middle_threshold)
    while column_position < len(original_matrix[middle_threshold]) - 1:
        row_position = top_threshold
        while row_position < bottom_threshold:
            if original_matrix[row_position][column_position] == 0:
                #print "checking  Removal part with row and position"
                #print(row_position)
                #print(column_position)
                fillAndDelete2(original_matrix, row_position, top_threshold, bottom_threshold, column_position)
                save_image(original_matrix, "1.jpeg")
            row_position = row_position + 1
        column_position = column_position + 1

    column_position = 0
    while column_position < len(original_matrix[middle_threshold]) - 1:
        row_position = top_threshold
        while row_position < bottom_threshold:
            if matrix_copy[row_position][column_position] == 0:
                (firstBoxLeft, firstBoxTop, firstBoxRight, firstBoxBottom, secondBoxLeft, secondBoxTop, secondBoxRight, secondBoxBottom) = calculateBox(matrix_copy, row_position, top_threshold, bottom_threshold, column_position)
                removePotentialLinePixelsInBox(original_matrix, firstBoxLeft, firstBoxTop, firstBoxRight, firstBoxBottom, matrix_copy)
                removePotentialLinePixelsInBox(original_matrix, secondBoxLeft, secondBoxTop, secondBoxRight, secondBoxBottom, matrix_copy)
                save_image(original_matrix, "2.jpeg")
            row_position = row_position + 1
        column_position = column_position + 1