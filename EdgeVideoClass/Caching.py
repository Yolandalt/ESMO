# 10个CNN模型，决定每个CNN模型是否存在6个边缘节点上（y = 0或者1）
import numpy as np

class ModelCaching:
    def __init__(self,
                 n_edge=6,  # MEC数量
                 n_CNNModel=8,  # CNN 模型数量
                 ):
        self.n_edge = n_edge
        self.n_CNNModel = n_CNNModel
        # self.caching_decision = 1 # 缓存决策（0或1），假设都缓存
        # 定义模型缓存决策,包含
        self.caching_decision_all = np.zeros((self.n_edge, self.n_CNNModel))
        # 初始化每个CNN模型所需要的缓存大小
        self.cache_require = np.random.uniform(1, 5, self.n_CNNModel)
        # 设定每个CNN模型可以提供的精度大小
        self.accuracy_provide = np.random.uniform(0.4, 1, self.n_CNNModel)

        # 初始化每个MEC节点的最大缓存空间
        self.cache_max_mec = np.random.uniform(5, 10, self.n_edge)
        # MEC节点上已经使用的缓存空间
        self.cache_mec = np.zeros(self.n_edge)
        # 定义模型上缓存的CNN模型数量
        self.num_CNN_in_edge = np.random.randint(0, n_CNNModel, self.n_edge)

    # 判断给点的边缘节点上的缓存资源是否还有空余
    def is_cache_full(self, edgeNode):
        # 考虑每个边缘节点上的模型缓存
        for j in range(self.n_CNNModel):
            if self.caching_decision_all[edgeNode][j] == 1:
                self.cache_mec[edgeNode] += self.cache_require[j]  # 如果已经缓存则添加
        # 遍历完所有模型之后，可以得到当前已经使用的缓存资源
        if self.cache_mec[edgeNode] <= self.cache_max_mec[edgeNode]:
            # 如果小于，还有空余，返回true
            return 1
        else:
            return 0

    # 初始化模型缓存决策，需要满足资源限制
    def init_caching_decision(self):
        # 对每一个MEC节点上的缓存决策进行操作
        for i in range(self.n_edge):
            # 判断节点上的缓存资源是否足够
            while self.is_cache_full(i):
                # 足够，则随机选择一个CNN模型进行缓存
                self.caching_decision_all[i][np.random.randint(0, self.n_CNNModel)] = 1
        return self.caching_decision_all











