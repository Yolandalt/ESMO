import cv2
import matplotlib.pyplot as plt
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
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

img1=cv2.imread("./1.jpg")
diff1=getdiff(img1)
print(diff1)
# print('img1:',getss(diff1))

img11=cv2.imread("./2.jpg")
diff11=getdiff(img11)
# print('img11:',getss(diff11))

img111=cv2.imread("./3.jpg")
diff111=getdiff(img111)
# print('img11:',getss(diff11))

ss1=getss(diff1)
ss2=getss(diff11)
ss3=getss(diff111)
# print("两张照片的方差为：%s"%(abs(ss1-ss2)))
x=range(30)
plt.figure("avg")
plt.plot(x,diff1,marker="*",color='#cc66ff', label="$Frame1$")
plt.plot(x,diff11,marker="o",color='#0066FF', label="$Frame2$")
plt.plot(x,diff111,marker="^",color='#FF3366',label="$Frame3$")

plt.legend(prop=font1)
plt.tick_params(labelsize=10)
plt.xlabel("The side length of the image", font2)
plt.ylabel("Average pixels in each row", font2)
plt.savefig("./similarity_result1.jpg")
plt.savefig("./similarity_result1.eps")
plt.savefig("./similarity_result1.pdf")

plt.legend()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()