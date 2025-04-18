import cv2
import numpy as np

#Get input from the user as a string
file = input("What file do you want to use? ")



#Read in intial image, set imread to 0 to ensure a gray scale image
image = cv2.imread(file)
image_gray = cv2.imread(file, 0)

#save shape of image
h,w,c = image.shape

image = image
sobel = image
canny = image

"""
In this problem, you will detect edges in an image by two methods:
(1)
Write your own Sobel filtering code in Python to detect edges, and
(2)
Use the Canny edge detector in OpenCV to detect edges.
Either way, take as input the color images, “cheerios,” “professor,” “gear,” and “circuit,” shown in Figure 2, and convert them to binary images showing edges in black on a white background.
The inner workings of the Canny edge detector are explained in the lecture and in the appendix at the end of this document.
Since the Canny results change significantly depending on specified parameters (threshold1, threshold2, apertureSize, and L2gradient), use some GUIs such as slider bars and radio buttons to allow a user to iteratively change and apply a different combination of parameters and see the result on the screen. Use your program to find the best combination of parameters for each of the four images.
For each image, compare and discuss the results of the Sobel filtering and the Canny edge detection.
"""

"""
binary images created by each method:
cheerios-sobel.png, cheerios-canny.png
professor-sobel.png, professor-canny.png
gear-sobel.png, gear-canny.png
circuit-sobel.png, circuit-sobel.png
"""

#Sobel Filter
#Derivative Filter + Gaussian Smoothing

#Sobel -> edge can be one less on each side is okay

#Sobel
    
#opencv_threshold

#opencv_guassian_blur

#Apply Gaussian Blur with 7 by 7 filter to smooth image before edge detection
Gauss = cv2.GaussianBlur(image_gray, (7,7), 0)

#Create Vertial and Horizontal Sobel Filter
Sob_Fil_V = (1/8)*np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Sob_Fil_H = (1/8)*np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

#Record height and width
h, w = image_gray.shape

#Save Gaussian image as i_g
i_g = Gauss

#function to apply filter 3x3 to pixel 3x3
def pixel_value(np1,np2):
        pixel = (np1[0][0]*np2[0][0])+(np1[0][1]*np2[0][1])+(np1[0][2]*np2[0][2])+(np1[1][0]*np2[1][0])+(np1[1][1]*np2[1][1])+(np1[1][2]*np2[1][2])+(np1[2][0]*np2[2][0])+(np1[2][1]*np2[2][1])+(np1[2][2]*np2[2][2])
        return pixel

#start list of final sobel filter
Sobel_list = []
#Range from 0 to h
for j in range(0,h):
    #Account for edges be not starting on the edge pixel at 0 and the end
    if j == 0:
        pass
    elif j == (h-1):
        pass
    else:
        #Start Sobel row list for final sobel edge image 
        Sobel_Row = []
        for i in range(0,w):
            #Account for edges be not starting on the edge pixel at 0 and the end
            if i == 0:
                pass
            elif i == (w-1):
                pass
            else:
                #Set of 3x3 pixels the filter will be applied to 
                pixel_array = np.array([[i_g[j-1][i-1],i_g[j-1][i],i_g[j-1][i+1]], 
                                        [i_g[j][i-1],i_g[j][i],i_g[j][i+1]], 
                                        [i_g[j+1][i-1], i_g[j+1][i], i_g[j+1][i+1]]])
                #Apply sobel vertical and horizontal filter using pixel_value function
                output_pixel_V = pixel_value(np1=Sob_Fil_V,np2=pixel_array)
                output_pixel_H = pixel_value(np1=Sob_Fil_H,np2=pixel_array)
                #get magnetude of vertical and horizontal filter
                output_pixel = (output_pixel_V**2 + output_pixel_H**2)**0.5
                #Append each new pixel to a row
                Sobel_Row.append(output_pixel)
        #Append each new row for the final image
        Sobel_list.append(Sobel_Row)
#Output Sobel image without threshold
Sobel = np.array(Sobel_list)
#Use opencv threshold with THRESH_BINARY_INV in order to make white image with black edges 
ret, Sobel_edge = cv2.threshold(Sobel, 3, 255, cv2.THRESH_BINARY_INV)

#cheerios.png threshold 12
#professor.png threshold 7
#gear.png threshold 12
#circuit.png threshold 3

#################################################################################################

#Create windows for Canny edge images
cv2.namedWindow("Output Canny", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Canny", w, h) 

#Create maximums for tracker bars
maxvalue = 1000
MaxApSize = 100
T_o_F = 1

#Save image as canny
canny = image

#Functions to print output of toggle bar 
def Up(track_val):
    print("Upper threshold")
    print(track_val)
def Low(track_val):
    print("Lower threshold")
    print(track_val)
def Apert(track_val):
    print("Aperture Size")
    print(track_val)
def Thres(track_val):
    print("L2 on or off")
    print(track_val)

#Create toggle bars for lower threshold, upper threshold, aperture size, and using L2 gradient
cv2.createTrackbar('Upper', 'Output Canny', 0, maxvalue, Up)
cv2.createTrackbar('Lower', 'Output Canny', 0, maxvalue, Low)
cv2.createTrackbar('ApertureSize', 'Output Canny', 0, MaxApSize, Apert)
cv2.createTrackbar("L2 gradient", 'Output Canny', 0, T_o_F, Thres)

#Generate Window and Change output image
while True:
    #Display New Image using cv2 imshow
    cv2.imshow('Output Canny', canny)
    #Record output of Trackbar Positions
    Lower = cv2.getTrackbarPos('Lower', 'Output Canny') 
    Upper = cv2.getTrackbarPos('Upper', 'Output Canny')
    ApertureSize = cv2.getTrackbarPos('ApertureSize', 'Output Canny') 
    L2_out = cv2.getTrackbarPos('L2 gradient', 'Output Canny')
    #Ensure L2 output is either True or False
    if L2_out == 0:
        L2 = False
        print("False")
    else:
        L2 = True
        print("True")
    #Wait 1millisecond before refreshing image
    k = cv2.waitKey(1)
    #Change out altered image with original image to avoid iteratively adding canny filters
    canny = image
    #Apply canny filter with inputs from tracker bars
    canny = cv2.Canny(canny, Upper, Lower, ApertureSize, L2gradient = L2)
    #Record the state of the Window displaying image and tracker bars, if this 
    #Window is closed record the event
    val = cv2.getWindowProperty('Output Canny', cv2.WND_PROP_VISIBLE)
    #Once the user is satified with the output and the  
    #window is closed the loop breaks
    if val == 0:
        print("exit")
        break

#Close all Windows
cv2.destroyAllWindows()

#Print out final outputs for record keeping
print("Final Upper")
print(Upper)
print("Final Lower")
print(Lower)
print("Final ApertureSize")
print(ApertureSize)
#Save the final color image as input-filename-color.png and display the file in the second window
#Create GUI for Canny Edge Detector
#OpenCV Trackbar GUI
#Cheerio Threshold 100

#Use opencv threshold with THRESH_BINARY_INV in order to make white image with black edges
ret, canny = cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY_INV)

#Output image window based on size of original image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", w, h) 
#Output Sobel Edge image window based on size of original image
cv2.namedWindow("Output Sobel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Sobel", w, h) 

#Output Canny Edge image window based on size of original image
cv2.namedWindow("Output Canny", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Canny", w, h) 

#Displays original image using windows generated above 
cv2.imshow("Original Image", image)

#Displays new image using windows generated above
cv2.imshow("Output Sobel", Sobel_edge)

#Displays new image using windows generated above
cv2.imshow("Output Canny", canny)

# fucntion the program waits for any key to be pressed
cv2.waitKey(0)
# The destoyAllWindows fucntion closes all windows after the script is complete.
cv2.destroyAllWindows()

#Split file on . for png and file name
newname = file.split(".")
#concatenate list of strings and add in name of file 
filesobel = newname[0]+"-sobel."+newname[1]
filecanny = newname[0]+"-canny."+newname[1]

#Use opencv's imwrite function to pass in file name and image that is being saved
cv2.imwrite(filesobel,Sobel_edge)
cv2.imwrite(filecanny,canny)
