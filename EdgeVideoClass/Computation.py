# 10个CNN模型，决定每个CNN模型是否存在6个边缘节点上（y = 0或者1）
import numpy as np
import math
import time

class ComputationOffload:
    def __init__(self,
                 # env,
                 n_edge=6,  # MEC数量
                 n_user=5,  # 用户数量
                 ):
        # self.env = env
        self.n_edge = n_edge
        self.n_user = n_user

        # 初始化每个边缘节点的计算能力
        self.freq_mec = np.random.uniform(10, 20, self.n_edge) * 1e9,  # MEC 计算单元频率

        # print(self.freq_mec)# 远程云计算能力(给每个frame的计算能力)
        # self.freq_cloud = 0.2e9
        self.freq_cloud = 100e9

        # 每个边缘节点上卸载的用户数
        # self.users_in_edge = []


    def get_cloud_computing_capacities(self):
        return self.freq_cloud










