import cv2
import numpy as np

#Get input from the user as a string
file = input("What file do you want to use? ")
#file = "pcb.png"
#file = "dog.png"
#file = "circuitboard.png"
file = "wedding.png"

#Read in intial image
image = cv2.imread(file)

#obtain shape of image
h,w,c = image.shape

#image = image
#improved = image

#Smooth then sharpen

"""
As discussed in class, the quality of an image can be improved by two types of area-to-pixel filters: 
(1) smoothing filters, and (2) sharpening filters. The former will reduce noise in an image, and the latter will make an image sharper.
In this problem, you are asked to find a smoothing filter, a sharpening filter, or a combination of them that improves the 
quality of each of the noisy and/or blurry images shown in Figure 1. Noisy/blurry images are shown on the left-hand side, and part of 
their ground truth images are shown on the right-hand side. 

Your task is to find a good combination of smoothing and sharpening filters and the order of applying them to improve each image 
and make it as close as possible to the ground truth.
You do not need to write your own filters for this problem. Instead, use any OpenCV smoothing functions, including cv2.blur(), 
cv2.boxFilter(), cv2.GaussianBlur(), cv2.medianBlur(), cv2.bilateralFilter(). For sharpening, you may use cv2.filter2D() with a sharpening kernel that you define. 
You may also create an “unsharp masking” effect by combining a Gaussian-smoothed image and the original image by using cv2.addWeighted().
"""

"""
improved images created by your program:
pcb-improved.png
golf-improved.png
pots-improved.png
rainbow-improved.png
"""
#ksize = (3,3)

#image = cv2.blur(src=image,ksize=ksize)
#cv2.boxFilter(), cv2.GaussianBlur(), cv2.medianBlur(), cv2.bilateralFilter(). For sharpening, you may use cv2.filter2D() with a sharpening kernel that you define. 
#You may also create an “unsharp masking” effect by combining a Gaussian-smoothed image and the original image by using cv2.addWeighted()
# Docs https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed

#cv2.blur 
#parameters image ksize 
#cv2.boxFilter(),
#parameters image, ksize, normalize
#cv2.GaussianBlur(),
#parameters image ksize sigma_x sigma_y takes x sigma if not given
#cv2.medianBlur(),
#parameters image ksize
#cv2.bilateralFilter()
#parameters image, d (Diameter of each pixel neighborhood), sigmaColor, sigmaSpace

#“unsharp masking”
#cv2.filter2D()
#parameters image, kernal (input array)
#cv2.addWeighted()
#parameters image, alpha, image2 beta, 0
#cv2.Laplacian()
#parameters image, ksize (intergers)

#Set unique parameters per image

#Sharpening Filter
    #kern = np.array([[0,0,0],
    #                 [0,1,0],
    #                 [0,0,0]])
    #improved = cv2.filter2D(src=blur, ddepth=-1, kernel=kern)

if file == "pcb.png":
    #Smoothing Filter
    #Use median filter to remove peak noise, using a 3x3 filter
    dots = cv2.medianBlur(image, ksize=3)
    #Sharpening Filter
    #Create Gaussian blurred version of image with removed noise
    #This blurred version of the image will be used for the 
    #Unsharp filter. The parameters for the Guassian is a 5 by 5 
    #filter with an XYSigma of 20 
    blur = cv2.GaussianBlur(dots, (5,5), 20)
    #blur = cv2.bilateralFilter(image, 5, 85, 85)
    #Sharpening Filter
    #Use addWeighted filter in combination of the Guassian filter to sharpen image
    improved = cv2.addWeighted(dots, 4.5, blur, -3.2,0)
    
    

elif file == "circuitboard.png":
    #Smoothing Filter
    #Create Gaussian blurred version of image with removed noise
    #This blurred version of the image will be used for the 
    #Unsharp filter. The parameters for the Guassian is a 5 by 5 
    #filter with an XYSigma of 20 
    blur = cv2.GaussianBlur(image, (5,5), 20)
    #blur = cv2.bilateralFilter(image, 5, 85, 85)
    #Sharpening Filter
    #Use addWeighted filter in combination of the Guassian filter to sharpen image
    improved = cv2.addWeighted(image, 4.5, blur, -3,0)


elif file == "dog.png":
    #Smoothing Filter
    #Use median filter to remove peak noise, using a 3x3 filter
    dots = cv2.medianBlur(image, ksize=3)
    #Sharpening Filter
    #Create Gaussian blurred version of image with removed noise
    #This blurred version of the image will be used for the 
    #Unsharp filter. The parameters for the Guassian is a 5 by 5 
    #filter with an XYSigma of 21
    blur = cv2.GaussianBlur(dots, (5,5), 21)
    #blur = cv2.bilateralFilter(image, 5, 85, 85)
    #Sharpening Filter
    #Use addWeighted filter in combination of the Guassian filter to sharpen image
    improved = cv2.addWeighted(dots, 4.5, blur, -3.6,0)
    #kern = np.array([[0,0,-1,0,0],
    #                 [0,-1,-1,-1,0],
    #                 [-1,-1,13,-1,-1],
    #                 [0,-1,-1,-1,0],
    #                 [0,0,-1,0,0]])
    #improved = cv2.filter2D(src=blur, ddepth=-1, kernel=kern)
    

elif file == "wedding.png":
    #Smoothing Filter
    #Use median filter to remove peak noise, using a 3x3 filter
    blur = cv2.medianBlur(image, ksize=3)
    #Sharpening Filter
    #sharpen image slightly using a filter with a kernal of [0,0,0][0,1,0][0,0,0]
    kern = np.array([[0,0,0],
                     [0,1,0],
                     [0,0,0]])
    #Apply filter to median image with 2D filter and kernal expressed above
    improved = cv2.filter2D(src=blur, ddepth=-1, kernel=kern)

else:
    print("File name not in program")


#Save the final color image as filename-improved.png and display the file in the second window

#Output image window based on size of original image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", int(w), int(h)) 
#Output color improved window based on size of original image
cv2.namedWindow("Output Improved Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Improved Image", int(w), int(h)) 

#Displays original image using windows generated above 
cv2.imshow("Original Image", image)

#Displays improved image using windows generated above
cv2.imshow("Output Improved Image", improved)

# fucntion the program waits for any key to be pressed
cv2.waitKey(0)
# The destoyAllWindows fucntion closes all windows after the script is complete.
cv2.destroyAllWindows()

#Split file on . for png and file name
newname = file.split(".")
#concatenate list of strings and add in name of file 
filecolor = newname[0]+"-improved."+newname[1]

#Use opencv's imwrite function to pass in file name and image that is being saved
cv2.imwrite(filecolor,improved)
