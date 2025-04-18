import cv2

file = "spade-terminal.png"

#Read in intial image
image = cv2.imread(file)

image_matched = image
h,w,c = image.shape

# Read image as grayscale image 
im = cv2.imread(file,cv2.IMREAD_GRAYSCALE)

# Threshold image 
_,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

for i in range(3):
    im = cv2.dilate(im, None)
for i in range(2):
    im = cv2.erode(im, None)

cont, hier = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cont_lst = []

for i in range(len(cont)):
    c = cont[i]
    area = cv2.contourArea(c)
    if area >= 7000 and area <= 9000:
        cont_lst.append(i)

Targ_Cont = 1

cont1 = cont[cont_lst[Targ_Cont]]
#image = cv2.drawContours(image, cont, cont_lst[Targ_Cont], (255,0,0),-1)


for i in range(29):
    if i == Targ_Cont:
        pass
    else:
        cont2 = cont[cont_lst[i]]
        ret12 = cv2.matchShapes(cont1, cont2,1,0.0)
        if ret12 > 0.24:
            image = cv2.drawContours(image, cont, cont_lst[i], (0,0,255),-1)
            print("error detected")
            print(ret12)

cv2.namedWindow("Spade Matching Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Spade Matching Original", int(w), int(h)) 

cv2.namedWindow("Spade Matching Complete", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Spade Matching Complete", int(w), int(h)) 

cv2.imshow("Spade Matching Original", image)
cv2.imshow("Spade Matching Complete", im)

cv2.waitKey(0)
cv2.destroyAllWindows()

newname = file.split(".")
fileMatched = newname[0]+"-output."+newname[1]

cv2.imwrite(fileMatched,image_matched)
cv2.imwrite("B-W-Spade.png", im)