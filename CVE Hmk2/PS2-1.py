#import cv2 and numpy libraries
import cv2
import numpy as np

#Get input from the user as a string
file = input("What file do you want to use? ")

#Read in intial image, set imread to 0 to ensure a gray scale image
gray_image = cv2.imread(file, 0)
#record h and w for window sizes
h,w = gray_image.shape

#Finds the maximum pixel in the image numpy array using np.amax 
maxpixel = np.amax(gray_image)
#Finds the minimum pixel in the image numpy array using np.amin
minpixel = np.amin(gray_image)

#Make a look-up table to convert the lowest gray value to blue and the highest gray value to red.
#The other gray values should be mapped to rainbow colors by the method explained in the lecture.

#Do not use OpenCV
#Hard coding tone curve graphs for blue, green, and red
#the scaled value is a value between 0 and 1 with 0 being the min and 1 being the max
#The if/elif statements for each tone curve corresponds to the values on the tone curve graph
#This ensures the values of the pixels shift from blue to red
def bluefunc(scaled):
    if scaled <= .25: 
        blue = 255
    elif (scaled > 0.25 and scaled < .5): 
        blue = ((-255/0.25)*(scaled-0.25))+255
    elif scaled >= .5:
        blue = 0
    else:
        #error has occurred as pixel is not between 0 and 255
        print("pixel out of scope")
        #ensure the pixel is an integer not a float
    return int(blue)
def greenfunc(scaled):
    if scaled <= .25: 
        green = (255/0.25)*(scaled)
    elif (scaled > 0.25 and scaled < .75): 
        green = 255
    elif scaled >= .75:
        green = ((-255/0.25)*(scaled-0.75))+255
    else:
        #error has occurred as pixel is not between 0 and 255
        print("pixel out of scope")
        #ensure the pixel is an integer not a float
    return int(green)
def redfunc(scaled):
    if scaled <= .5: 
        red = 0
    elif (scaled > 0.5 and scaled < 0.75): 
        red = (255/0.25)*(scaled-0.5)
    elif scaled >= .75:
        red = 255
    else:
        #error has occurred as pixel is not between 0 and 255
        print("pixel out of scope")
        #ensure the pixel is an integar not a float
    return int(red)

#scaling the pixel values from 0 to 1 for mapping to tone curve
#The function (pixel-minpixel)/(maxvalue-minvalue) scales each pixel from zero to one
# This way if the minimum pixel isn't exactly 0 or the max pixel isn't exactly 255 then 
# this scales the pixels so the max is now 255 and the min is now 0 

#scales the pixels in the image from 0 to 1 to ensure the maximum value is red and minimum pixel value is blue
def scalepix(pixelvalue, minvalue=minpixel, maxvalue=maxpixel):
    scaled = (pixelvalue-minvalue)/(maxvalue-minvalue) 
    tablelookup = int(scaled*255)
    if (tablelookup < 0 or tablelookup > 255):
        print("Out of scope")
        print(tablelookup)
    return tablelookup

# For the tone curve, a table is created that takes in a value from 0 to 255, this curve is built with the function above 
# using (bluefunc, greenfunc, redfunc) to get the exact pixel value. This maps an individual gray scale
# pixel to a color pixel on the tone curve  

#Creates lookup table to be used to color image
table = []
for value in list(range(0,256)):
    scaled=value/255
    blue = bluefunc(scaled)
    green = greenfunc(scaled)
    red = redfunc(scaled)
    colorpixel = [blue, green, red]
    table.append(colorpixel)

table = np.array(table)

#changing out pixels from gray scale to tone curve using the lookup table and the scaled function created above for the color map
#this iterates through every pixel scales it and looks up the value on the lookuptable above to individually change out pixels

def lookuptable(image, minpix, maxpix):
    color_image = []
    for i in image:
        pixel = []
        for j in i:
            pixel.append(table[scalepix(pixelvalue = j, minvalue=minpix, maxvalue=maxpix)])
        color_image.append(pixel)
            
    return color_image

#uses lookup table function above to generate final image
colorimage = np.array(lookuptable(image=gray_image, minpix=minpixel, maxpix=maxpixel))
#ensures the numpy array is readable to opencv by making the numpy array an array of 1-byte unsigned integers 
colorimage = colorimage.astype(np.uint8)


#Using OpenCV functions, draw a cross in a circle to indicate the pixel of the highest gray value.
#Draw the cross and circle with white. If multiple pixels share the same highest gray value, place the
#cross and circle at the center of gravity of these pixels. Figure 4 shows a sample input image and
#output image.

#Use OpenCV cv2.line() and cv2.circle()
#Find location of max pixel(s) by using np.where to identify location of max pixels equal to the value identified
#by zipping and converting it to a list the output is easier to use for find the x and y center of gravity

#Get center of gravity
location = list(zip(*np.where(gray_image==maxpixel)))
#Create list of x and y values of max pixels
xlist = []
ylist = []

#unwrap tuples and add them to x and y lists
for tuples in location:
    y, x = tuples
    xlist.append(x)
    ylist.append(y)

#Obtain the x center of gravity of the max value pixels by summing all the pixels x values and dividing by the number of pixels
#Repeat the same process for y 
#combine in a tuple to get center of gravity
cgx = sum(xlist)/len(xlist)
cgy = sum(ylist)/len(ylist)
#ensures values are an integer not a float
cg = (int(cgx),int(cgy))

#set size for lines
size = 15

#draws an cross on the center of gravity of an image 

#Create starting and end points for the line by add and subtracting from the center of gravity coordinates with size
cgs1 = (int(cgx+size),int(cgy))
cge1 = (int(cgx-size),int(cgy))
cgs2 = (int(cgx),int(cgy+size))
cge2 = (int(cgx),int(cgy-size))

#Draws a circle using OpenCV's circle function, the radius is the size of the circle
#image is the numpy array being used to draw the circle
#cg is the center of the circle
# thickness is the thickness of the line to draw the circle and
# color is the color channel pixel value, in this case (255,255,255) is the value for white
cv2.circle(colorimage, cg, radius=10, color=(255,255,255), thickness=2)
#Draws a cross using OpenCV's line function
#image is the numpy array being used to draw the cross
#cgs and cge are the starting and ending coordinates for the line
#thickness is the thickness of the line to draw the cross and
# color is the color channel pixel value, in this case (255,255,255) is the value for white
cv2.line(colorimage, cgs1, cge1, color=(255,255,255), thickness=1)
cv2.line(colorimage, cgs2, cge2, color=(255,255,255), thickness=1)

#Save the final color image as input-filename-color.png and display the file in the second window

#Output grayscale image window based on size of original image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", w, h) 
#Output color image window based on size of original image
cv2.namedWindow("Output color image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output color image", w, h) 

#Displays original image using windows generated above 
cv2.imshow("Original Image", gray_image)

#Displays new image using windows generated above
cv2.imshow("Output color image", colorimage)

# fucntion the program waits for any key to be pressed
cv2.waitKey(0)
# The destoyAllWindows fucntion closes all windows after the script is complete.
cv2.destroyAllWindows()

#Split file on . for png and file name
newname = file.split(".")
#concatenate list of strings and add in name of file 
filecolor = newname[0]+"-color."+newname[1]

#Use opencv's imwrite function to pass in file name and image that is being saved
cv2.imwrite(filecolor,colorimage)