import cv2
import numpy as np 
import open3d as o3d

#Use Open3D
fileL = input("Select Left Image\n")
fileR = input("Select Right Image\n")
fileL = fileL + "-left.png"
fileR = fileR + "-right.png"

#Read in intial image
imageL = cv2.imread(fileL)
imageR = cv2.imread(fileR)

h,w,c = imageL.shape
h,w,c = imageR.shape

# Read image as grayscale image 
im_L = cv2.imread(fileL,cv2.IMREAD_GRAYSCALE)
im_R = cv2.imread(fileR,cv2.IMREAD_GRAYSCALE)

if fileL == "rdesatni-left.png": 
    h = 555
    w = 620
    imageL = cv2.resize(imageL,(620,555))
    imageR = cv2.resize(imageR,(620,555))
    im_L = cv2.resize(im_L,(620,555))
    im_R = cv2.resize(im_R,(620,555))

cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Disparity", int(w), int(h))

def nothing(x):
    pass

cv2.createTrackbar("minDisparity", 'Disparity', 1, 100, nothing) 
cv2.createTrackbar("numDisparities", 'Disparity', 2, 100, nothing) 
cv2.createTrackbar("blockSize", 'Disparity', 2, 11, nothing) 
cv2.createTrackbar('P1', 'Disparity', 1, 300, nothing) 
cv2.createTrackbar('P2', 'Disparity', 1, 900, nothing) 
cv2.createTrackbar('disp12MaxDiff', 'Disparity', 1, 200, nothing)
cv2.createTrackbar('uniquenessRatio', 'Disparity', 1, 100, nothing) 
cv2.createTrackbar('speckleWindowSize', 'Disparity', 1, 100, nothing) 
cv2.createTrackbar('speckleRange', 'Disparity', 1, 100, nothing) 
cv2.createTrackbar('preFilterCap', 'Disparity', 1, 63, nothing) 

numDisparities_T = 16
blockSize_T = 15
stereo = cv2.StereoSGBM_create(numDisparities=numDisparities_T, 
                               blockSize=blockSize_T, 
                               )
imageD = stereo.compute(im_L, im_R)

while True:
    cv2.imshow('Disparity', imageD)
    minDisparity_T = cv2.getTrackbarPos("minDisparity", 'Disparity')
    numDisparities_T = cv2.getTrackbarPos("numDisparities", 'Disparity')
    numDisparities_T = numDisparities_T*16
    blockSize_T = cv2.getTrackbarPos("blockSize", 'Disparity')
    P1_T = cv2.getTrackbarPos("P1", 'Disparity')
    P2_T = cv2.getTrackbarPos("P2", 'Disparity')
    disp12MaxDiff_T = cv2.getTrackbarPos("disp12MaxDiff", 'Disparity')
    uniquenessRatio_T = cv2.getTrackbarPos("uniquenessRatio", 'Disparity')
    speckleWindowSize_T = cv2.getTrackbarPos("speckleWindowSize", 'Disparity')
    speckleRange_T = cv2.getTrackbarPos("speckleRange", 'Disparity')
    preFilterCap_T = cv2.getTrackbarPos("preFilterCap", 'Disparity')
    stereo = cv2.StereoSGBM_create(minDisparity=minDisparity_T,
                                numDisparities=numDisparities_T, 
                                blockSize=blockSize_T, 
                                P1=P1_T,P2=P2_T,
                                disp12MaxDiff=disp12MaxDiff_T,
                                uniquenessRatio=uniquenessRatio_T,
                                speckleWindowSize=speckleWindowSize_T,
                                speckleRange=speckleRange_T,
                                preFilterCap=preFilterCap_T,
                                )
    imageD = stereo.compute(im_L, im_R)
    #Normalize Image D
    Beta = 255
    Alpha = 0
    imageD = cv2.normalize(imageD,Alpha,Beta,norm_type=cv2.NORM_MINMAX)
    imageD = imageD.astype(np.uint8,copy=False)
    k = cv2.waitKey(1)
    
    val = cv2.getWindowProperty('Disparity', cv2.WND_PROP_VISIBLE)
    if val == 0:
        print("exit")
        print("minDisparity")
        print(minDisparity_T)
        print("numDisparities")
        print(numDisparities_T)
        print("blockSize")
        print(blockSize_T)
        print("P1")
        print(P1_T)
        print("P2")
        print(P2_T)
        print("disp12MaxDiff")
        print(disp12MaxDiff_T)
        print("uniquenessRatio")
        print(uniquenessRatio_T)
        print("speckleWindowSize")
        print(speckleWindowSize_T)
        print("speckleRange")
        print(speckleRange_T)
        print("preFilterCap")
        print(preFilterCap_T)
        break

cv2.destroyAllWindows()

cv2.namedWindow("Image Left", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image Left", int(w), int(h)) 

cv2.namedWindow("Image Right", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image Right", int(w), int(h)) 

cv2.imshow("Image Left", imageL)
cv2.imshow("Image Right", imageR)
cv2.imshow("Disparity", imageD)

cv2.waitKey(0)
cv2.destroyAllWindows()

newname = fileR.split("-")
File_Disparity = newname[0]+"-disparity.png"

cv2.imwrite(File_Disparity,imageD)

scale = 55000

points = np.empty((h,w,3))
for v in range(h):
    for u in range(w):
        D = imageD[v,u]
        X = u - w/2
        Y = -v - h/2
        Z = -scale/D
        points[v, u, :] = [X, Y, Z]

mask = imageD > imageD.min()
ply_3D = points[mask]
colors = imageL[mask]

#ps7-tips
file3D = open(newname[0] + ".ply", "w")
file3D.write("ply\n")
file3D.write("format ascii 1.0\n")
file3D.write(f"element vertex {ply_3D.shape[0]}\n")
file3D.write("property float32 x\n")
file3D.write("property float32 y\n")
file3D.write("property float32 z\n")
file3D.write("property uint8 red\n")
file3D.write("property uint8 green\n")
file3D.write("property uint8 blue\n")
file3D.write("end_header\n")

print(ply_3D.shape)

for i in range(ply_3D.shape[0]):
    file3D.write(f"{ply_3D[i,0]:.3f} {ply_3D[i,1]:.3f} {ply_3D[i,2]:.3f} {colors[i,2]} {colors[i,1]} {colors[i,0]}\n")
file3D.close()

ply_file2 = o3d.io.read_point_cloud(newname[0] + ".ply")
o3d.visualization.draw_geometries([ply_file2])