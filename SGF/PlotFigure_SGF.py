import random
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

Similarity = []
# read data
dir = "./Results_SGF_two.txt"
Similarity = np.loadtxt(dir)

# 999 images
T_th = [0.1, 0.5, 5, 10, 15, 50, 100, 300,1000,3000]
Num_pic_all_T = [0]*len(T_th)
# The greater the similarity (the greater the variance, the less similar)
t = 0
for th in range(len(T_th)):
    print("11111111111")
    print(T_th[th])
    i = 0
    j = 0
    Num_pic_all_T[th] += 1  # saving the first frame
    while i <len(Similarity) and j <len(Similarity):
        print("i"+str(i))
        print("j" + str(i))
        j = i+1
        while j < len(Similarity):
            if j >= len(Similarity):
               continue
            if Similarity[i][j]/100 < T_th[th]:  # not similar
                Num_pic_all_T[th] += 1  # saving second frame
                # remember index
                i = j
                break
            j += 1
dir = "./Results_SGF_final_plot.txt"
np.savetxt(dir, Num_pic_all_T)

print(Num_pic_all_T)
x1_NUM = T_th
x1 = list(range(len(x1_NUM)))

y = Num_pic_all_T

total_width1, n1 = 2, 3
width1 = total_width1 / n1
plt.bar(x1, y, width=width1, color='#75bbfd', hatch='\\', alpha=0.7)
for i in range(len(x1)):
    x1[i] += width1
for i in range(len(x1)):
    x1[i] = x1[i] - width1
plt.xticks(x1)
plt.xticks(x1, x1_NUM)

plt.tick_params(labelsize=10)
plt.xlabel('The value of similarity threshold', font2)
plt.ylabel('The number of video frames', font2)

plt.savefig("./result_SGF.jpg")
plt.savefig("./result_SGF.eps")
plt.savefig("./result_SGF.pdf")
plt.show()

