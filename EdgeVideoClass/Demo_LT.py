import numpy as np
import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
# 3D绘图示意
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
print(len(x))

a = x+3*y
print(len(a))


