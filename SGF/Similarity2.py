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

def getss(list):     # Calculate the average
    avg=sum(list)/len(list)     # Define the variance variable ss, the initial value is 0
    ss=0     # Calculate variance
    for l in list:
        ss+=(l-avg)*(l-avg)/len(list)     # Return variance
    return ss

# Get the average value of pixels in each row
def getdiff(img):     # Define side length
    Sidelength=30     # Zoom image
    img=cv2.resize(img,(Sidelength,Sidelength),interpolation=cv2.INTER_CUBIC)     # Grayscale processing
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # The list saves the average value of each row of pixels
    avglist=[]     # Calculate the average of each row and save it to the list
    for i in range(Sidelength):
        avg=sum(gray[i])/len(gray[i])
        avglist.append(avg)     # Returns the average value
    return avglist

img1=cv2.imread("./1.jpg")
diff1=getdiff(img1)
print(diff1)

img11=cv2.imread("./2.jpg")
diff11=getdiff(img11)

img111=cv2.imread("./3.jpg")
diff111=getdiff(img111)

ss1=getss(diff1)
ss2=getss(diff11)
ss3=getss(diff111)

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