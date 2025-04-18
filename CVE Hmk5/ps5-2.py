import cv2  
import numpy as np
import pandas as pd
import imutils


#Get input from the user as a string
file = input("What image do you want to use?")
file = "blood-cells1.jpg"
#file = "blood-cells2.png"
#file = "blood-cells3.png"

#Read in intial image
image = cv2.imread(file)
image_g = cv2.imread(file,0)

image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = (255,255,255))
image_g = cv2.copyMakeBorder(image_g, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 255)
#obtain shape of image
h,w,c = image.shape

scale1 = 0.32
scale2 = 0.27
scale3 = 0.3
area1 = 400
area2 = 400
area3 = 1000
arealst = []
ctlst = []
momentlst = []
centerlst = []
p_angle = []
box_size = []
crop = []
hlist=[]
wlist=[]
majorlist = []
minorlist = []

if file == "blood-cells1.jpg":
    #(1) Convert the three blood-cell images to binary black-and-white images, 
    gray = image_g.copy()
    image_g = cv2.medianBlur(image_g,9)
    blur = cv2.GaussianBlur(image_g, (5,5), 20)
    image_g = cv2.addWeighted(image_g, 1.29, blur, -0.2,0)
    AG = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    binary = AG 
    #(2) detect blobs,caliculate blob sizes and principal axes, 
    #blobs = image
    k_e = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    D1 = cv2.dilate(binary,k_e,iterations=1)
    E1 = cv2.erode(D1,k_e,iterations=3)
    blobs = E1
    cont, hier = cv2.findContours(blobs,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image_c = image.copy()
    #(3) and generate images called blood-cells1-catalog.jpg, blood-cells2-catalog.jpg, and blood-cells3-catalog.jpg that list all 
    #the detected blobs in order of size, as shown in Figure 4. In your catalog images, the major axis of each blob should be horizontal.
    ig, size, array = hier.shape
    for contour in range(size):
        area = cv2.contourArea(cont[contour])
        if area >= 10000:
            print("ignore border")
            #print(area)
        elif area >=area1:
            arealst.append(area)
            ctlst.append(cont[contour])
            contours = cv2.drawContours(image_c,[cont[contour]],-1,(0,255,0),3)
            moments =  cv2.moments(cont[contour])
            momentlst.append(moments)
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cv2.circle(image_c, center,radius=1,color=(0,0,255),thickness=5)
            centerlst.append(center)
            e = cv2.fitEllipse(cont[contour])
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            x1 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] + 90) * np.pi / 180.0)))
            y1 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] + 90) * np.pi / 180.0)))
            x2 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] - 90) * np.pi / 180.0)))
            y2 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] - 90) * np.pi / 180.0)))
            angle = e[2]
            major = e[1][1]
            minor = e[1][0]
            p_angle.append(angle)
            cv2.line(image_c, (x1, y1), (x2, y2), (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(cont[contour])
            box_size.append([x,y,w,h])
            img = cv2.rectangle(image_c, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_crop = binary[y:(y+h), x:(x+w)]
            hlist.append(h)
            wlist.append(w)
            crop.append(img_crop)
            majorlist.append(major)
            minorlist.append(minor)
        else:
            print("not a blob")
    
    hnum = np.array(hlist)
    wnum = np.array(wlist)
    hmax = np.max(hnum)
    wmax = np.max(wnum)
    blobs_cata = contours
    Contour_Dict = {'Area':arealst,'Contour':ctlst, 'Moment':momentlst, 'Center':centerlst, 'Box Size':box_size, 'P_angle':p_angle, 'Crop':crop, 'Major':majorlist,'Minor':minorlist}
    Contour_DF = pd.DataFrame(data=Contour_Dict)
    Contour_DF.sort_values(by=['Major'], ascending=False, inplace=True, ignore_index=True)
    Contour_ser = pd.Series(Contour_DF['Contour'])
    Crop_ser = pd.Series(Contour_DF['Crop'])
    Angle_ser = pd.Series(Contour_DF['P_angle'])
    Maj_ser = pd.Series(Contour_DF['Major'])
    Min_ser = pd.Series(Contour_DF['Minor'])
    
    Cell_list = []
    
    for cell in range(len(arealst)):
        Single_Cell = Crop_ser[cell]
        P_angle = Angle_ser[cell]
        delta_angle = 90 - P_angle
        height, width = Single_Cell.shape[:2]
        center = (width/2, height/2)
        Single_Cell_invert = cv2.bitwise_not(Single_Cell)
        Single_Cell_Rot = imutils.rotate_bound(Single_Cell_invert,delta_angle)
        Single_Cell_Rot_invert = cv2.bitwise_not(Single_Cell_Rot)
        Single_Cell_Half = cv2.resize(Single_Cell_Rot_invert, (0,0), fx=scale1, fy=scale1)
        Cell_list.append(Single_Cell_Half)
    
    table = np.zeros_like(blobs_cata)
    catalog = cv2.bitwise_not(table)
    h_c,w_c,c = catalog.shape
    h_c = h_c
    w_c = w_c
    num_line = int(np.ceil(len(arealst)**0.5))
    num_line_r = (len(arealst)**0.5)
    for grid in range(num_line):
        x_val = (int(w_c/num_line))*grid
        startpoint = (x_val,0)
        endpoint = (x_val,h_c)
        catalog = cv2.line(catalog,startpoint,endpoint,(0,0,0),1)
    for grid in range(num_line):
        y_val = (int(h_c/num_line))*grid
        startpoint = (0,y_val)
        endpoint = (w_c,y_val)
        catalog = cv2.line(catalog,startpoint,endpoint,(0,0,0),1)
    catalog = cv2.cvtColor(catalog, cv2.COLOR_BGR2GRAY)
    iter = 0
    for cell_r in range(num_line):
        for cell_c in range(num_line):
            x_val = (int(w_c/num_line))
            y_val = (int(h_c/num_line))
            pad = 3
            x_offset = cell_c*x_val+pad
            y_offset = cell_r*y_val+pad  
            h, w = Cell_list[iter].shape
            catalog[y_offset:y_offset+h, x_offset:x_offset+w] = Cell_list[iter]
            iter = iter + 1
            if iter >= len(arealst):
                break
        if iter >= len(arealst):
            break


elif file == "blood-cells2.png":
    gray = image_g.copy()
    image_g = cv2.bilateralFilter(image_g,30,40,10)
    blur = cv2.GaussianBlur(image_g, (7,7), 30)
    image_g = cv2.addWeighted(image_g, 2.59, blur, -1.5,0)
    AG = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    binary = AG
    k_e = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    D1 = cv2.dilate(binary,k_e,iterations=1)
    E1 = cv2.erode(D1,k_e,iterations=2)
    blobs = E1.copy()
    cont, hier = cv2.findContours(blobs,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image_c = image.copy()
    ig, size, array = hier.shape

    for contour in range(size):
        area = cv2.contourArea(cont[contour])
        if area >= 10000:
            print("ignore border")
        elif area >=area2:
            arealst.append(area)
            ctlst.append(cont[contour])
            contours = cv2.drawContours(image_c,[cont[contour]],-1,(0,255,0),3)
            moments =  cv2.moments(cont[contour])
            momentlst.append(moments)
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cv2.circle(image_c, center,radius=1,color=(0,0,255),thickness=5)
            centerlst.append(center)
            e = cv2.fitEllipse(cont[contour])
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            x1 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] + 90) * np.pi / 180.0)))
            y1 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] + 90) * np.pi / 180.0)))
            x2 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] - 90) * np.pi / 180.0)))
            y2 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] - 90) * np.pi / 180.0)))
            angle = e[2]
            major = e[1][1]
            minor = e[1][0]
            p_angle.append(angle)
            cv2.line(image_c, (x1, y1), (x2, y2), (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(cont[contour])
            box_size.append([x,y,w,h])
            img = cv2.rectangle(image_c, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_crop = binary[y:(y+h), x:(x+w)]
            hlist.append(h)
            wlist.append(w)
            crop.append(img_crop)
            majorlist.append(major)
            minorlist.append(minor)
        else:
            print("not a blob")

    blobs_cata = contours
    Contour_Dict = {'Area':arealst,'Contour':ctlst, 'Moment':momentlst, 'Center':centerlst, 'Box Size':box_size, 
                    'P_angle':p_angle, 'Crop':crop, 'Major':majorlist,'Minor':minorlist}
    Contour_DF = pd.DataFrame(data=Contour_Dict)
    Contour_DF.sort_values(by=['Major'], ascending=False, inplace=True, ignore_index=True)
    Contour_ser = pd.Series(Contour_DF['Contour'])
    Crop_ser = pd.Series(Contour_DF['Crop'])
    Angle_ser = pd.Series(Contour_DF['P_angle'])
    Maj_ser = pd.Series(Contour_DF['Major'])
    Min_ser = pd.Series(Contour_DF['Minor'])
    Cell_list = []
    for cell in range(len(arealst)):
        Single_Cell = Crop_ser[cell]
        P_angle = Angle_ser[cell]
        delta_angle = 90 - P_angle

        height, width = Single_Cell.shape[:2]
        center = (width/2, height/2)
        
        Single_Cell_invert = cv2.bitwise_not(Single_Cell)
        Single_Cell_Rot = imutils.rotate_bound(Single_Cell_invert,delta_angle)
        Single_Cell_Rot_invert = cv2.bitwise_not(Single_Cell_Rot)
        Single_Cell_Half = cv2.resize(Single_Cell_Rot_invert, (0,0), fx=scale2, fy=scale2)
        Cell_list.append(Single_Cell_Half)
    
    table = np.zeros_like(blobs_cata)
    catalog = cv2.bitwise_not(table)
    h_c,w_c,c = catalog.shape
    h_c = h_c
    w_c = w_c
    num_line = int(np.ceil(len(arealst)**0.5))
    num_line_r = (len(arealst)**0.5)
    for grid in range(num_line):
        x_val = (int(w_c/num_line))*grid
        startpoint = (x_val,0)
        endpoint = (x_val,h_c)
        catalog = cv2.line(catalog,startpoint,endpoint,(0,0,0),1)
    for grid in range(num_line):
        y_val = (int(h_c/num_line))*grid
        startpoint = (0,y_val)
        endpoint = (w_c,y_val)
        catalog = cv2.line(catalog,startpoint,endpoint,(0,0,0),1)
    catalog = cv2.cvtColor(catalog, cv2.COLOR_BGR2GRAY)
    iter = 0
    for cell_r in range(num_line):
        for cell_c in range(num_line):
            x_val = (int(w_c/num_line))
            y_val = (int(h_c/num_line))
            pad = 3
            x_offset = cell_c*x_val+pad
            y_offset = cell_r*y_val+pad  
            h, w = Cell_list[iter].shape
            catalog[y_offset:y_offset+h, x_offset:x_offset+w] = Cell_list[iter]
            iter = iter + 1
            if iter >= len(arealst):
                break
        if iter >= len(arealst):
            break

elif file == "blood-cells3.png":
    gray = image_g.copy()
    image_g = cv2.bilateralFilter(image_g,8,16,8) #10,20,10
    blur = cv2.GaussianBlur(image_g, (7,7), 30)
    image_g = cv2.addWeighted(image_g, 1.9, blur, -0.8, 0)
    AG = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    binary = AG
    k_e = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    D1 = cv2.dilate(binary,k_e,iterations=1)
    E1 = cv2.erode(D1,k_e,iterations=2)
    blobs = E1.copy()
    cont, hier = cv2.findContours(blobs,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image_c = image.copy()
    ig, size, array = hier.shape

    for contour in range(size):
        area = cv2.contourArea(cont[contour])
        if area >= 10000:
            print("ignore border")
        elif area >=area3:
            arealst.append(area)
            ctlst.append(cont[contour])
            contours = cv2.drawContours(image_c,[cont[contour]],-1,(0,255,0),3)
            moments =  cv2.moments(cont[contour])
            momentlst.append(moments)
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cv2.circle(image_c, center,radius=1,color=(0,0,255),thickness=5)
            centerlst.append(center)
            e = cv2.fitEllipse(cont[contour])
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            x1 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] + 90) * np.pi / 180.0)))
            y1 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] + 90) * np.pi / 180.0)))
            x2 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] - 90) * np.pi / 180.0)))
            y2 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] - 90) * np.pi / 180.0)))
            angle = e[2]
            major = e[1][1]
            minor = e[1][0]
            p_angle.append(angle)
            cv2.line(image_c, (x1, y1), (x2, y2), (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(cont[contour])
            box_size.append([x,y,w,h])
            img = cv2.rectangle(image_c, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_crop = binary[y:(y+h), x:(x+w)]
            hlist.append(h)
            wlist.append(w)
            crop.append(img_crop)
            majorlist.append(major)
            minorlist.append(minor)
        else:
            print("not a blob")
    
    blobs_cata = contours
    Contour_Dict = {'Area':arealst,'Contour':ctlst, 'Moment':momentlst, 'Center':centerlst, 'Box Size':box_size, 
                    'P_angle':p_angle, 'Crop':crop, 'Major':majorlist,'Minor':minorlist}
    Contour_DF = pd.DataFrame(data=Contour_Dict)
    Contour_DF.sort_values(by=['Major'], ascending=False, inplace=True, ignore_index=True)
    Contour_ser = pd.Series(Contour_DF['Contour'])
    Crop_ser = pd.Series(Contour_DF['Crop'])
    Angle_ser = pd.Series(Contour_DF['P_angle'])
    Maj_ser = pd.Series(Contour_DF['Major'])
    Min_ser = pd.Series(Contour_DF['Minor'])
    Cell_list = []
    for cell in range(len(arealst)):
        Single_Cell = Crop_ser[cell]
        P_angle = Angle_ser[cell]
        delta_angle = 90 - P_angle
        height, width = Single_Cell.shape[:2]
        center = (width/2, height/2)
        Single_Cell_invert = cv2.bitwise_not(Single_Cell)
        Single_Cell_Rot = imutils.rotate_bound(Single_Cell_invert,delta_angle)
        Single_Cell_Rot_invert = cv2.bitwise_not(Single_Cell_Rot)
        Single_Cell_Half = cv2.resize(Single_Cell_Rot_invert, (0,0), fx=scale3, fy=scale3)
        Cell_list.append(Single_Cell_Half)

    table = np.zeros_like(blobs_cata)
    catalog = cv2.bitwise_not(table)
    h_c,w_c,c = catalog.shape
    h_c = h_c
    w_c = w_c
    num_line = int(np.ceil(len(arealst)**0.5))
    num_line_r = (len(arealst)**0.5)
    for grid in range(num_line):
        x_val = (int(w_c/num_line))*grid
        startpoint = (x_val,0)
        endpoint = (x_val,h_c)
        catalog = cv2.line(catalog,startpoint,endpoint,(0,0,0),1)
    for grid in range(num_line):
        y_val = (int(h_c/num_line))*grid
        startpoint = (0,y_val)
        endpoint = (w_c,y_val)
        catalog = cv2.line(catalog,startpoint,endpoint,(0,0,0),1)
    catalog = cv2.cvtColor(catalog, cv2.COLOR_BGR2GRAY)
    iter = 0
    for cell_r in range(num_line):
        for cell_c in range(num_line):
            x_val = (int(w_c/num_line))
            y_val = (int(h_c/num_line))
            pad = 3
            x_offset = cell_c*x_val+pad
            y_offset = cell_r*y_val+pad  
            h, w = Cell_list[iter].shape
            catalog[y_offset:y_offset+h, x_offset:x_offset+w] = Cell_list[iter]
            iter = iter + 1
            if iter >= len(arealst):
                break
        if iter >= len(arealst):
            break
else:
    print("file not recognized")

#Output image window based on size of original image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", int(w), int(h)) 
#Output color improved window based on size of original image
cv2.namedWindow("Gray", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gray", int(w), int(h)) 
cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Binary", int(w), int(h)) 
cv2.namedWindow("Blobs", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blobs", int(w), int(h))
cv2.namedWindow("Catalog", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Catalog", int(w), int(h)) 
#Displays original image using windows generated above 
cv2.imshow("Original Image", image)
cv2.imshow("Gray", gray)
cv2.imshow("Binary", binary)
cv2.imshow("Blobs", blobs_cata)
cv2.imshow("Catalog", catalog)
# fucntion the program waits for any key to be pressed
cv2.waitKey(0)
# The destoyAllWindows fucntion closes all windows after the script is complete.
cv2.destroyAllWindows()
#Split file on . for png and file name
newname = file.split(".")
#concatenate list of strings and add in name of file 
filegray = newname[0]+"-gray."+newname[1]
filebinary = newname[0]+"-binary."+newname[1]
fileblobs = newname[0]+"-blobs."+newname[1]
fileblobs_cata = newname[0]+"-blobs_cata."+newname[1]
filecatalog = newname[0]+"-catalog."+newname[1]
#Use opencv's imwrite function to pass in file name and image that is being saved
cv2.imwrite(filegray,gray)
cv2.imwrite(filebinary,binary)
cv2.imwrite(fileblobs,blobs)
cv2.imwrite(fileblobs_cata,blobs_cata)
cv2.imwrite(filecatalog,catalog)