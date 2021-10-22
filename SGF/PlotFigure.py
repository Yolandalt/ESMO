import random
import numpy as np
import matplotlib.pyplot as plt

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

Similarity = []
# 读数据
# with open("./Results_SGF.txt", "r") as f:
#     lines = f.readlines()
#     list1 = lines[0].split(",")
#     a = list1.pop()
#     for item in list1:
#         if item:
#             Similarity.append(float(item))

# print(Similarity)
# 阈值设置

# 一共999张图片
T_th = [0, 50, 100, 150, 500, 1000, 3000]
# T_th = [3000, 1000, 500, 150, 100, 50, 0]
# Num_pic_all_T = np.zeros(len(T_th))
# # 相似度越大（方差越大，越不相似）
# for th in range(len(T_th)):
#     for i in range(len(Similarity)):
#         if Similarity[i] > T_th[th]: # 有变化
#             Num_pic_all_T[th] += 1


x1_NUM = T_th
x1 = list(range(len(x1_NUM)))
y = [376.0,518.0,518.0,694.0,822.0,908.0,998.0]

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
plt.tick_params(labelsize=10.5)
plt.xlabel('The value of similarity threshold', font2)
plt.ylabel('The number of video frames', font2)

plt.savefig("./result_SGF.jpg")
plt.savefig("./result_SGF.eps")
plt.savefig("./result_SGF.pdf")
plt.show()

