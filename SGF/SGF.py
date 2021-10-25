import cv2
import os
import json
import numpy as np
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }

# computing average value
def getss(list):
    avg=sum(list)/len(list)     # Define the variance variable ss, the initial value is 0
    ss=0     # Calculating variance
    for l in list:
        ss+=(l-avg)*(l-avg)/len(list)     # return variance
    return ss

# Get the average value of pixels in each row
def getdiff(img):     # Define side length
    Sidelength=30     # Zoom image
    img=cv2.resize(img,(Sidelength,Sidelength),interpolation=cv2.INTER_CUBIC)     # Grayscale processing
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # The list saves the average value of each row of pixels
    avglist=[]     # Calculate the average of each row and save it to the list
    for i in range(Sidelength):
        avg=sum(gray[i])/len(gray[i])
        avglist.append(avg)     # return average
    return avglist


dir_all=[]
filePath = './archive/train/'
for i in os.walk(filePath):
    dir_all = i[2]
print(dir_all)
img1 = 0
diff1 = 0
img11 = 0
diff11 = 0
simirarity = np.zeros((len(dir_all),len(dir_all)))

for i in range(len(dir_all)):
    img1 = cv2.imread("./archive/train/"+dir_all[i])
    diff1 = getdiff(img1)
    ss1 = getss(diff1)
    for j in range(len(dir_all)):
        img11 = cv2.imread("./archive/train/"+dir_all[j])
        diff11 = getdiff(img11)
        ss2 = getss(diff11)
        s = 0
        if abs(ss1-ss2) == 0:
            s = 0
        else:
            s = round(1/abs(ss1-ss2),6)*10000
        simirarity[i][j] = s

dir = "./Results_SGF_two.txt"
np.savetxt(dir, simirarity)

cv2.waitKey(0)
cv2.destroyAllWindows()