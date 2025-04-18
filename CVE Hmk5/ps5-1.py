import cv2  
import numpy as np
import random

#Get input from the user as a string
#file = input("What image do you want to use?")
#file = "wall1.png"
file = "wall2.png"
#file = "wall1-original.png"
#file = "wall2-original.png"
#Read in intial image
image = cv2.imread(file)
h,w,c = image.shape

image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = (255,255,255)) 

if file == "wall1.png":
    #(1) Apply dilation and erosion operations to the input image to turn black regions into clean blobs. 
    #Name the new image, “wall1-blobs.png,” and save the file,
    #-> erosion and dialation techques to everything
    k_e = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    E1 = cv2.erode(image,k_e,iterations=2)
    D1 = cv2.dilate(E1,k_e,iterations=1)
    E2 = cv2.erode(D1,k_e,iterations=3)
    D1 = cv2.dilate(E2,k_e,iterations=1)
    #################################################
    blobs = D1
    blobs_f = blobs.copy()
    blobs_t = blobs.copy()
    #(2) Detect blobs and their contours; create an image that shows blob contours with randomly assigned 
    #colors, name the image,”wall1-contours.png,” and save the file,
    random_color = list(range(256))
    image_gray = cv2.cvtColor(blobs,cv2.COLOR_BGR2GRAY)
    cont, hier = cv2.findContours(image_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    ig, size, array = hier.shape
    for contour in range(size):
        area = cv2.contourArea(cont[contour])
        if area >= 20000:
            print("ignore crack")
        else:
            R = random.choice(random_color)
            G = random.choice(random_color)
            B = random.choice(random_color)
            contours = cv2.drawContours(blobs,[cont[contour]],-1,(B,G,R),3)
            if (area >= 2000 and area < 20000):
                print("probably a crack")
                save_contour = [cont[contour]]
                print(area)
                
    #(3) Define your own thresholds for detecting cracks, and apply the thresholds to the blobs to keep 
    #only the blobs that are likely to be actual cracks,
    #(4) For each blob that is likely to be an actual crack, find its central axis by thinning operations. 
    #Create an image that shows the central axis, name the image, “wall1-cracks,png,” and save the file,
    cracks = np.zeros_like(blobs_t)
    cracks = cv2.bitwise_not(cracks)
    cv2.drawContours(cracks, save_contour, 0, (255, 255, 255), 1)
    cv2.fillPoly(cracks, save_contour, (0, 0, 0))
    crack_t = cracks.copy() 
    cracks = cv2.dilate(crack_t,k_e,iterations=4)
    # Important: Reverse image (black background)
    #img1 = cv2.bitwise_not(crack_t)
    # Kernel: 4 neighbor
    #k_e = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # Target image
    #thin = np.zeros(img1.shape, dtype=np.uint8)
    # repeat until no white area
    #i = 0 
    #while cv2.countNonZero(img1) != 0:
    
    """
    while True:
        er = cv2.erode(img1, k_e)
    # OPEN: erosion then dilation (remove noise)
        op = cv2.morphologyEx(er, cv2.MORPH_OPEN,k_e)
        subset = er - op
        thin = cv2.bitwise_or(subset, thin)
        #img1 = er.copy()
        img1 = thin
        i = i + 1
        if i >= 1:
            break
    img1 = cv2.bitwise_not(img1)
    cracks = img1   
    """

elif file == "wall2.png":
    save_contour = []
    k_e = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    E1 = cv2.erode(image,k_e,iterations=1)
    D1 = cv2.dilate(E1,k_e,iterations=2)
    E2 = cv2.erode(D1,k_e,iterations=2)

    D1 = cv2.dilate(E2,k_e,iterations=1)

    blobs = D1
    blobs_f = blobs.copy()
    blobs_t = blobs.copy()

    random_color = list(range(256))
    image_gray = cv2.cvtColor(blobs,cv2.COLOR_BGR2GRAY)
    cont, hier = cv2.findContours(image_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    print("Information of hier")
    print(hier.shape)
    ig, size, array = hier.shape
    for contour in range(size):
        area = cv2.contourArea(cont[contour])
        if area >= 20000:
            print("ignore crack")
        else:
            R = random.choice(random_color)
            G = random.choice(random_color)
            B = random.choice(random_color)
            contours = cv2.drawContours(blobs,[cont[contour]],-1,(B,G,R),3)

            if (area >= 1500 and area < 20000):
                print("probably a crack")
                save_contour.append([cont[contour]])
                print(area)

    cracks = np.zeros_like(blobs_t)
    cracks = cv2.bitwise_not(cracks)
    for CT_lst in range(len(save_contour)):
        cv2.drawContours(cracks, save_contour[CT_lst], 0, (255, 255, 255), 1)
        cv2.fillPoly(cracks, save_contour[CT_lst], (0, 0, 0))
    crack_t = cracks.copy() 
    cracks = cv2.dilate(crack_t,k_e,iterations=3)

    """
    # Important: Reverse image (black background)
    img1 = cv2.bitwise_not(crack_t)
    # Kernel: 4 neighbor
    k_e = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # Target image
    thin = np.zeros(img1.shape, dtype=np.uint8)
    # repeat until no white area
    i = 0 
    #while cv2.countNonZero(img1) != 0:
    while True:
        er = cv2.erode(img1, k_e)
    # OPEN: erosion then dilation (remove noise)
        op = cv2.morphologyEx(er, cv2.MORPH_OPEN,k_e)
        subset = er - op
        thin = cv2.bitwise_or(subset, thin)
        img1 = er.copy()
        i = i + 1
        if i >= 3:
            break
    img1 = cv2.bitwise_not(img1)
    cracks = img1 
    """

else:
    print("file not recognized")

#Output image window based on size of original image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", int(w), int(h)) 

cv2.namedWindow("Blobs", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blobs", int(w), int(h)) 

cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contours", int(w), int(h)) 

cv2.namedWindow("Cracks", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cracks", int(w), int(h)) 

#Displays image using windows generated above 
cv2.imshow("Original Image", image)

#Displays image using windows generated above
cv2.imshow("Blobs", blobs_f)
cv2.imshow("Contours", contours)
cv2.imshow("Cracks", cracks)

# fucntion the program waits for any key to be pressed
cv2.waitKey(0)
# The destoyAllWindows fucntion closes all windows after the script is complete.
cv2.destroyAllWindows()

#Split file on . for png and file name
newname = file.split(".")
#concatenate list of strings and add in name of file 
fileblob = newname[0]+"-blobs."+newname[1]
filecontour = newname[0]+"-contours."+newname[1]
filecracks = newname[0]+"-cracks."+newname[1]
#Use opencv's imwrite function to pass in file name and image that is being saved
cv2.imwrite(fileblob,blobs_f)
cv2.imwrite(filecontour,contours)
cv2.imwrite(filecracks,cracks)