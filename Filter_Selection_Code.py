import cv2
import numpy as np

#Get input from the user as a string
file = input("What file do you want to use? ")
file = "blood-cells3.png"
#file = "blood-cells2.png"
#file = "blood-cells1.jpg"
#file = "pcb.png"
#file = "dog.png"
#file = "circuitboard.png"
# file = "wedding.png"


#Read in intial image, set imread to 0 to ensure a gray scale image
image = cv2.imread(file)
image_gray = cv2.imread(file, 0)

h,w,c = image.shape

image = image

cv2.namedWindow("Output Blur", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Blur", w, h+200)

#cv2.namedWindow("Output Filter", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Output Filter", w, h) 

T_o_F = 1

if file == "pcb.png":
    image = cv2.medianBlur(image, ksize=5)

if file == "dog.png":
    image = cv2.medianBlur(image, ksize=3)

filter = image
blur = image

def nothing(x):
    pass

cv2.createTrackbar("Guassian Blur", 'Output Blur', 0, T_o_F, nothing)
cv2.createTrackbar("Bilateral Blur", 'Output Blur', 0, T_o_F, nothing)
cv2.createTrackbar("Median Blur", 'Output Blur', 0, T_o_F, nothing)

cv2.createTrackbar('ksize', 'Output Blur', 0, 10, nothing)
cv2.createTrackbar('d', 'Output Blur', 15, 100, nothing)
cv2.createTrackbar('Color Sigma', 'Output Blur', 75, 100, nothing)
cv2.createTrackbar('Space Sigma', 'Output Blur', 75, 100, nothing)
cv2.createTrackbar('XY Sigma', 'Output Blur', 1, 100, nothing)

#blur

while True:
    cv2.imshow('Output Blur', blur)
    val = cv2.getWindowProperty('Output Blur', cv2.WND_PROP_VISIBLE)
    print("val")
    print(val)
    if val == 0:
        print("exit")
        break

    blur = image
    blur1 = image

    G_blur = cv2.getTrackbarPos("Guassian Blur", 'Output Blur')
    if G_blur == 1:
        ksize = cv2.getTrackbarPos("ksize", 'Output Blur')
        ksize = 2*ksize + 1 
        ksizet = (ksize,ksize)
        XYSigma = cv2.getTrackbarPos("XY Sigma", 'Output Blur')
        k = cv2.waitKey(1)
        blur = image
        blur = cv2.GaussianBlur(blur, ksizet, XYSigma)
        print("Gaussian Filter applied")
        val = cv2.getWindowProperty('Output Blur', cv2.WND_PROP_VISIBLE)
        if val == 0:
            print("exit")
            break
    
    B_blur = cv2.getTrackbarPos("Bilateral Blur", 'Output Blur') 
    if B_blur == 1:
        d = cv2.getTrackbarPos('d', 'Output Blur')
        color = cv2.getTrackbarPos("Color Sigma", 'Output Blur')
        space = cv2.getTrackbarPos("Space Sigma", 'Output Blur')
        k = cv2.waitKey(1)
        blur1 = image
        if d <= 1:
            d = 2
        blur = cv2.bilateralFilter(blur1, d, color, space)
        #if blur1 == blur:
        #    print("No Change")
        print("Bilaterial Filter Applied")
        val = cv2.getWindowProperty('Output Blur', cv2.WND_PROP_VISIBLE)
        if val == 0:
            print("exit")
            break
    
    M_blur = cv2.getTrackbarPos("Median Blur", 'Output Blur')
    if M_blur == 1:
        ksize = cv2.getTrackbarPos("ksize", 'Output Blur')
        ksize = 2*ksize + 1 
        k = cv2.waitKey(1)
        blur = image
        blur = cv2.medianBlur(blur, ksize=ksize)
        print("Median Filter applied")
        val = cv2.getWindowProperty('Output Blur', cv2.WND_PROP_VISIBLE)
        if val == 0:
            print("exit")
            break
    
    if (M_blur == 0 and B_blur == 0 and G_blur == 0):
        k = cv2.waitKey(1)
        blur = image
        #print("No Change")
        val = cv2.getWindowProperty('Output Blur', cv2.WND_PROP_VISIBLE)
        if val == 0:
            print("exit")
            break

    else:
        print("Nothing")

    val = cv2.getWindowProperty('Output Blur', cv2.WND_PROP_VISIBLE)
    if val == 0:
        print("exit")
        break

cv2.destroyAllWindows()
FinalBlur = blur
TestBlur = blur 

#Sharpen

cv2.namedWindow("Output Filter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Filter", w, h+200)

cv2.createTrackbar("Filter 2D Sharpen", 'Output Filter', 0, T_o_F, nothing)
cv2.createTrackbar("Filter 2D Laplacian", 'Output Filter', 0, T_o_F, nothing)
cv2.createTrackbar("Filter 2D Laplacian2", 'Output Filter', 0, T_o_F, nothing)
cv2.createTrackbar("Filter Unsharp Mask", 'Output Filter', 0, T_o_F, nothing)

cv2.createTrackbar('Center', 'Output Filter', 0, 40, nothing)
cv2.createTrackbar('Rim', 'Output Filter', 1, 10, nothing)
cv2.createTrackbar('Middle', 'Output Filter', 1, 10, nothing)
cv2.createTrackbar('Alpha', 'Output Filter', 1, 96, nothing)
cv2.createTrackbar('Beta', 'Output Filter', 1, 96, nothing)

#blur
filter = TestBlur
while True:
    cv2.imshow('Output Filter', filter)
    filter = TestBlur
    Rim = 0
    Cent = 0
    Midd = 0
    val = cv2.getWindowProperty('Output Filter', cv2.WND_PROP_VISIBLE)
    if val == 0:
        print("exit")
        break

    D2_Sharp = cv2.getTrackbarPos("Filter 2D Sharpen", 'Output Filter')
    
    if D2_Sharp == 1:
        Cent = cv2.getTrackbarPos("Center", 'Output Filter')
        Rim_Scaled = cv2.getTrackbarPos("Rim", 'Output Filter')
        Rim_S = np.arange(-10,10)
        Rim = Rim_S[Rim_Scaled]
        kern = np.array([[Rim,Rim,Rim],[Rim,Cent,Rim],[Rim,Rim,Rim]])
        print(kern)
        k = cv2.waitKey(1)
        filter = blur
        filter = cv2.filter2D(src=TestBlur, ddepth=-1, kernel=kern)
        print("Filter 2D Sharpen applied")
    
    Lap = cv2.getTrackbarPos("Filter 2D Laplacian", 'Output Filter') 
    
    if Lap == 1:
        Cent = cv2.getTrackbarPos("Center", 'Output Filter')
        Rim_Scaled = cv2.getTrackbarPos("Rim", 'Output Filter')
        Rim_S = np.arange(-10,10)
        Rim = Rim_S[Rim_Scaled]
        Midd_Scaled = cv2.getTrackbarPos("Middle", 'Output Filter')
        Midd_S = np.arange(-10,10)
        Midd = Midd_S[Midd_Scaled]
        kern = np.array([[Rim,Midd,Rim],[Midd,Cent,Midd],[Rim,Midd,Rim]])
        print(kern)
        k = cv2.waitKey(1)
        filter = blur
        filter = cv2.filter2D(src=TestBlur,  ddepth=-1, kernel=kern)
        print("Filter 2D Laplacian Applied")

    Lap2 = cv2.getTrackbarPos("Filter 2D Laplacian2", 'Output Filter') 
    
    if Lap2 == 1:
        Cent = cv2.getTrackbarPos("Center", 'Output Filter')
        Rim_Scaled = cv2.getTrackbarPos("Rim", 'Output Filter')
        Rim_S = np.arange(-10,10)
        Rim = Rim_S[Rim_Scaled]
        Midd_Scaled = cv2.getTrackbarPos("Middle", 'Output Filter')
        Midd_S = np.arange(-10,10)
        Midd = Midd_S[Midd_Scaled]
        kern = np.array([[Rim,Rim,Midd,Rim,Rim],[Rim,Midd,Midd,Midd,Rim],[Midd,Midd,Cent,Midd,Midd],[Rim,Midd,Midd,Midd,Rim],[Rim,Rim,Midd,Rim,Rim]])
        print(kern)
        k = cv2.waitKey(1)
        filter = blur
        filter = cv2.filter2D(src=TestBlur,  ddepth=-1, kernel=kern)
        print("Filter 2D Laplacian Applied")

    UnMask = cv2.getTrackbarPos("Filter Unsharp Mask", 'Output Filter')
    
    if UnMask == 1:
        print("UnMask")
        Alpha_Scaled = cv2.getTrackbarPos("Alpha", 'Output Filter')
        Beta_Scaled = cv2.getTrackbarPos("Beta", 'Output Filter')
        Scaled = np.arange(-5,5,0.1)
        Alpha = Scaled[Alpha_Scaled]
        Beta = Scaled[Beta_Scaled]
        print(Alpha)
        print(Beta)
        k = cv2.waitKey(1)
        filter = blur
        filter = cv2.addWeighted(image, Alpha, TestBlur, Beta,0)
        print("Filter 2D Sharpen applied")
    
    Lap = cv2.getTrackbarPos("Filter 2D Laplacian", 'Output Filter') 
    
    if (Lap == 0 and Lap == 0 and D2_Sharp == 0):
        k = cv2.waitKey(1)
        blur = image
        #print("No Change")

    else:
        print("Nothing")

    val = cv2.getWindowProperty('Output Filter', cv2.WND_PROP_VISIBLE)
    if val == 0:
        print("exit")
        break

cv2.destroyAllWindows()

Finalfilter = filter
Testfilter = filter

#Weight unsharp mask
"""
cv2.namedWindow("Output Unsharp Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Unsharp Mask", w, h+200)

cv2.createTrackbar('Alpha', 'Output Filter', 0, 40, nothing)
cv2.createTrackbar('Beta', 'Output Filter', 1, 20, nothing)

while True:
    cv2.imshow('Output Unsharp Mask', image)
    val = cv2.getWindowProperty('Output Filter', cv2.WND_PROP_VISIBLE)
    if val == 0:
        print("exit")
        break
    Alpha = cv2.getTrackbarPos("Alpha", 'Output Unsharp Mask')
    Beta = cv2.getTrackbarPos("Beta", 'Output Unsharp Mask')
"""


"""
while True:
    cv2.imshow('Output filter', filter)
    Lower = cv2.getTrackbarPos('Lower', 'Output filter') 
    Upper = cv2.getTrackbarPos('Upper', 'Output filter')
    ApertureSize = cv2.getTrackbarPos('ApertureSize', 'Output filter') 



    k = cv2.waitKey(1)
    filter = image
    filter = cv2.Canny(filter, Upper, Lower, ApertureSize, L2gradient = L2)
    val = cv2.getWindowProperty('Output filter', cv2.WND_PROP_VISIBLE)
    if val == 0:
        print("exit")
        break

cv2.destroyAllWindows()
"""

#Save the final color image as input-filename-color.png and display the file in the second window

#Create GUI for Canny Edge Detector
#OpenCV Trackbar GUI

#Output grayscale image window based on size of original image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", w, h) 
#Output color image window based on size of original image
cv2.namedWindow("Output Blur", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Blur", w, h) 

#Output color image window based on size of original image
cv2.namedWindow("Output Filter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Filter", w, h) 

#Displays original image using windows generated above 
cv2.imshow("Original Image", image)
#cv2.imshow("Output Blur", blur)
cv2.imshow("Output Blur", FinalBlur)
cv2.imshow("Output Filter", filter)

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
cv2.imwrite(filecanny,filter)