import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
import cv2
import pandas as pd
#########################################################################################################

#Choose File
file1 = "Fish_V2_L.jpg"
file2 = "Fish_V2_R.jpg"


img1 = cv2.imread(file1)
print(img1.shape)
img2 = cv2.imread(file2)
print(img2.shape)
h1,w1,c = img1.shape
h2,w2,c = img1.shape
s = 4
hs1 = int(h1/s)
ws1 = int(w1/s)
hs2 = int(h2/s)
ws2 = int(w2/s)
img1 = cv2.resize(img1,(ws1,hs1))
img2 = cv2.resize(img2,(ws2,hs2))
#Output image window based on size of original image
cv2.namedWindow("Flat Fisheye Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Flat Fisheye Image", int(ws1), int(hs1))
#Output image window based on size of original image
cv2.namedWindow("Original Fisheye Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Fisheye Image", int(ws1), int(hs1))

def nothing(x):
    pass

cv2.createTrackbar("fx", "Flat Fisheye Image", 2944, 5000, nothing) 
cv2.createTrackbar('fy', "Flat Fisheye Image", 2209, 5000, nothing)
cv2.createTrackbar('k1', "Flat Fisheye Image", 0, 4989, nothing) 
cv2.createTrackbar('k2', "Flat Fisheye Image", 0, 4989, nothing) 
cv2.createTrackbar('k3', "Flat Fisheye Image", 0, 4998, nothing) 
cv2.createTrackbar('k4', "Flat Fisheye Image", 0, 4989, nothing)

pix_5a_x = 450
pix_5a_y = 450
fx = pix_5a_x 
fy = pix_5a_y 
cx = int(ws1/2)
cy = int(hs1/2)
k1 = -0.05
k2 = 0.02
k3 = -0.01
k4 = 0.002
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float64)
D = np.array([k1,k2,k3,k4], dtype=np.float64)
new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (ws1, hs1), np.eye(3), balance=1.0)
flat1 = cv2.fisheye.undistortImage(img1,K,D, None, new_camera_matrix)
k1range = np.arange(-5.0, -0.01, 0.001)
k2range = np.arange(0.01, 5.0, 0.001)
k3range = np.arange(-5.0, -0.001, 0.001)
k4range = np.arange(0.001, 0.50, 0.0001)
while True:
    cv2.imshow('Flat Fisheye Image', flat1)
    cv2.imshow('Original Fisheye Image',img1)
    fx = cv2.getTrackbarPos("fx", 'Flat Fisheye Image')
    fy = cv2.getTrackbarPos("fy", 'Flat Fisheye Image')
    k1 = cv2.getTrackbarPos("k1", 'Flat Fisheye Image')
    k1 = k1range[k1]
    k2 = cv2.getTrackbarPos("k2", 'Flat Fisheye Image')
    k2 = k2range[k2]
    k3 = cv2.getTrackbarPos("k3", 'Flat Fisheye Image')
    k3 = k3range[k3]
    k4 = cv2.getTrackbarPos("k4", 'Flat Fisheye Image')
    k4 = k4range[k4]
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float64)
    D = np.array([k1,k2,k3,k4], dtype=np.float64)
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (ws1, hs1), np.eye(3), balance=1.0)
    flat1 = cv2.fisheye.undistortImage(img1,K,D, None, new_camera_matrix)
    flat2 = cv2.fisheye.undistortImage(img2,K,D, None, new_camera_matrix)
    k = cv2.waitKey(1)
    val = cv2.getWindowProperty('Flat Fisheye Image', cv2.WND_PROP_VISIBLE)
    if val == 0:
        print("fx")
        print(fx)
        print("fy")
        print(fy)
        print("k1")
        print(k1)
        print("k2")
        print(k2)
        print("k3")
        print(k3)
        print("k4")
        print(k4)
        break

flat1_c = flat1[175:(hs1-175),275:(ws1-275)]
flat2_c = flat2[175:(hs1-175),275:(ws1-275)]

#Output image window based on size of original image
cv2.namedWindow("Original Fisheye Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Fisheye Image", int(ws1), int(hs1)) 
cv2.namedWindow("Flat Fisheye Image 1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Flat Fisheye Image 1", int(ws1), int(hs1)) 
cv2.namedWindow("Flat Fisheye Image 2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Flat Fisheye Image 2", int(ws1), int(hs1)) 
#Displays original image using windows generated above
 
cv2.imshow("Original Fisheye Image", img1)
cv2.imshow("Flat Fisheye Image 1", flat1_c)
cv2.imshow("Flat Fisheye Image 2", flat2_c)

# fucntion the program waits for any key to be pressed
cv2.waitKey(0)
# The destoyAllWindows fucntion closes all windows after the script is complete.
cv2.destroyAllWindows()
#Split file on . for png and file name
newname1 = file1.split(".")
newname2 = file2.split(".")
#concatenate list of strings and add in name of file 
fisheye_corrected1 = newname1[0]+"-flat."+ newname1[1]
fisheye_corrected2 = newname2[0]+"-flat."+ newname1[1]
#Use opencv's imwrite function to pass in file name and image that is being saved
print(fisheye_corrected1)
cv2.imwrite(fisheye_corrected1,flat1_c)
cv2.imwrite(fisheye_corrected2,flat2_c)


# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html 
MIN_MATCH_COUNT = 2

# images that works
# imgL = cv2.imread("fisheye-photo-L.jpg")          
# imgR = cv2.imread("fisheye-photo-R.jpg") 

# for combining part 1 code
imgR = cv2.imread(fisheye_corrected1)          
imgL = cv2.imread(fisheye_corrected2) 

# newer images, but doesn't quite work?
# imgL = cv2.imread("Fish_V3_L-flat.jpg")          
# imgR = cv2.imread("Fish_V3_R-flat.jpg")          

ln = imgL.copy()
rn = imgR.copy()

ln_gray = cv2.cvtColor(ln, cv2.COLOR_BGR2GRAY)
rn_gray = cv2.cvtColor(rn, cv2.COLOR_BGR2GRAY)

ln_RGB = cv2.cvtColor(ln, cv2.COLOR_BGR2RGB)
rn_RGB = cv2.cvtColor(rn, cv2.COLOR_BGR2RGB)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(ln_gray,None)        # kp, des = keypoints, descriptors
kp2, des2 = sift.detectAndCompute(rn_gray,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.

good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    print('what is m.queryIdx... ', kp1[m.queryIdx].pt)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    print('source and result points ', src_pts, dst_pts)
    # M is the matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)   
    matchesMask = mask.ravel().tolist()

    h,w = ln_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None 

warped_image = cv2.warpPerspective(imgL, M, (w,h))
dst = cv2.warpPerspective(imgL,M,(imgR.shape[1]+imgL.shape[1],imgR.shape[0]))
cv2.imshow('before dst', dst)
dst[0:imgR.shape[0], 0:imgR.shape[1]]=imgR
                                  
# Visualize the result
cv2.imshow('dst', dst)
cv2.imwrite('Restore_Sift.jpg', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(ln_RGB,kp1,rn_RGB,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

#########################################################################################################
model = YOLO("yolov8s.pt")  
"""
model.train(
    data="C:/Users/bibhu/OneDrive/Desktop/24678/Project/dataset/dataset.yaml",  # Dataset config
    epochs=50,  # Number of epochs
    imgsz=640,  # Image size
    batch=16,  # Batch size
    name="updated_model"  # Run name
)
val_results = model.val(
    data="C:/Users/bibhu/OneDrive/Desktop/24678/Project/dataset/dataset.yaml",  save=True )
    """
# Load the trained YOLOv8 model
model = YOLO("weights/best.pt")
# Run inference on the test image
results = model.predict(source="Restore_Sift.jpg",  save=True)

image_path = "Restore_Sift.jpg"
image = cv2.imread(image_path)
# Path to save the bounding box data
output_file = "C:/Users/rdesa/bounding_boxes_full_coordinates.txt"

# Save bounding box data to a file
with open(output_file, "w") as f:
    for result in results:
        f.write(f"Image: {result.path}\n") 
        for box in result.boxes:  
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to numerical values
            conf = box.conf[0].item()  # Confidence score as a numerical value
            cls = int(box.cls[0].item())  # Class ID as a numerical value
            # Calculate full corner coordinates
            top_left = (x1, y1)
            top_right = (x2, y1)
            bottom_left = (x1, y2)
            bottom_right = (x2, y2)
            f.write(f"Class ID: {cls}, Confidence: {conf:.2f}\n")
            f.write(f"Top-left: {top_left}, Top-right: {top_right}, Bottom-left: {bottom_left}, Bottom-right: {bottom_right}\n")
        f.write("\n")

# Custom visualization of bounding boxes without confidence scores
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
        cls = int(box.cls[0].item())  # Class ID
        label = model.names[cls]  # Get class label

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
output_path = "predict_final.jpg" #"C:/Users/rdesa/predict_custom.jpg"
cv2.imwrite(output_path, image)

# Define the input file
file_path = "bounding_boxes_full_coordinates.txt"

# Initialize an empty string to store the output
output_string = ""

# Read and parse the file
with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        if line.startswith("Top-left:"):
            top_left = tuple(map(float, line.split("(")[1].split(")")[0].split(", ")))
            bottom_right = tuple(map(float, line.split("(")[4].split(")")[0].split(", ")))
            output_string += f"{top_left[0]}, {top_left[1]}, {bottom_right[0]}, {bottom_right[1]}\n"

# Format the output as a Python string
output_string = f'"""\n{output_string.strip()}\n"""'

# Print the output
print(output_string)

# Bounding box data (x_min, y_min, x_max, y_max) - as raw input
bounding_boxes_raw = output_string

# Parse bounding box coordinates
bounding_boxes = []
for line in bounding_boxes_raw.strip().split("\n"):
    if line == '"""':
        pass
    else:
        coords = [float(coord) for coord in line.split(",")]
        bounding_boxes.append({
            "x_min": int(coords[0]),
            "y_min": int(coords[1]),
            "x_max": int(coords[2]),
            "y_max": int(coords[3])
        })

# Set dimensions for the binary map (adjust based on the use case)
map_dimensions = (1374, 1218)  # Width x Height flipped for the new system
binary_map = np.zeros(map_dimensions, dtype=int)

# Mark bounding boxes as obstacles in the binary map
for bbox in bounding_boxes:
    x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

    # Fill the binary map for the adjusted bounding box
    binary_map[x_min:x_max, y_min:y_max] = 1

# Save the binary map to a CSV file
output_path = "generated_binary_map.csv"
pd.DataFrame(binary_map).to_csv(output_path, header=False, index=False)

print(f"Binary map saved to {output_path}")