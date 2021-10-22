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
# 读数据
# with open("./Results_SGF_two.txt", "r") as f:
#     lines = f.readlines()
#     list1 = lines[0].split(",")
#     a = list1.pop()
#     for item in list1:
#         if item:
#             Similarity.append(float(item))
dir = "./Results_SGF_two.txt"
Similarity = np.loadtxt(dir)
# a = np.amax(Similarity)
# print(a)
# print(Similarity)
# 阈值设置




# 一共999张图片
T_th = [0.1, 0.5, 5, 10, 15, 50, 100, 300,1000,3000]
# T_th = [ 150, 3000]
# T_th = [3000, 1000, 500, 150, 100, 50, 0]
Num_pic_all_T = [0]*len(T_th)
# 相似度越大（方差越大，越不相似）
t = 0
for th in range(len(T_th)):
    print("11111111111")
    print(T_th[th])
    i = 0
    j = 0
    Num_pic_all_T[th] += 1  # 存第一张
    while i <len(Similarity) and j <len(Similarity):
        print("i"+str(i))
        print("j" + str(i))
        j = i+1
        while j < len(Similarity):
            if j >= len(Similarity):
               continue
            if Similarity[i][j]/100 < T_th[th]:  # 不相似
                Num_pic_all_T[th] += 1  # 存第二张
                # 记住index
                i = j
                break
            j += 1
            # t += 1
            # print("t" + str(i))
dir = "./Results_SGF_final_plot.txt"
np.savetxt(dir, Num_pic_all_T)

print(Num_pic_all_T)
x1_NUM = T_th
x1 = list(range(len(x1_NUM)))
# y = [376.0,518.0,518.0,694.0,822.0,908.0,998.0]
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

# plt.legend(prop=font1)
plt.tick_params(labelsize=10)
plt.xlabel('The value of similarity threshold', font2)
plt.ylabel('The number of video frames', font2)

plt.savefig("./result_SGF.jpg")
plt.savefig("./result_SGF.eps")
plt.savefig("./result_SGF.pdf")
plt.show()

