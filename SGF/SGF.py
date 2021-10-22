import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
def getss(list):     #计算平均值
    avg=sum(list)/len(list)     #定义方差变量ss，初值为0
    ss=0     #计算方差
    for l in list:
        ss+=(l-avg)*(l-avg)/len(list)     #返回方差
    return ss

#获取每行像素平均值
def getdiff(img):     #定义边长
    Sidelength=30     #缩放图像
    img=cv2.resize(img,(Sidelength,Sidelength),interpolation=cv2.INTER_CUBIC)     #灰度处理
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #avglist列表保存每行像素平均值
    avglist=[]     #计算每行均值，保存到avglist列表
    for i in range(Sidelength):
        avg=sum(gray[i])/len(gray[i])
        avglist.append(avg)     #返回avglist平均值
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

# print(type(simirarity))

dir = "./Results_SGF_two.txt" # 通过扩展名指定文件存储的数据为json格式
np.savetxt(dir, simirarity)
# dataset = np.loadtxt(dir)
# print(dataset)
# print(type(dataset))

# f = open(dir, "w")
# for i in range(len(simirarity)):
#     for j in range(len(simirarity)):
#         f.write(str(round(item, 2)) + ",")
# f.close()
# 阈值设置

cv2.waitKey(0)
cv2.destroyAllWindows()