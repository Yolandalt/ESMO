import numpy as np

class User_class:
    def __init__(self,
                 task_arrival_rate=0.5,  # 任务个数到达率——遵循泊松分布
                 energy_unit=5e-5,  #发送功率能量单元，和任务大小有关
                 n_edge = 6
                  ):
        self.task_arrival_rate = task_arrival_rate
        self.energy_unit = energy_unit
        self.n_edge = n_edge
        # 每个用户的frame数量
        self.n_frames = np.random.randint(1, 10)
        # 每个frame的bit数
        self.task_size = np.random.randint(300*1e3, 500*1e3, self.n_frames),  # 任务大小(每个任务有多少个bit)
        self. CPU_clock_per_bit = np.random.uniform(500, 900),  # cycle/bit 每个bit需要的CPU转数（w）

        self.frame_id = np.random.randint(0, 100, self.n_frames)
        self.user_capability = 1.2 * 1e9/self.n_frames  # 用户的本地计算能力
        # 本地应该加排队时延

        # self.user_location = []  # 位置初始化，用经度纬度表示
        # self.user_location_list_ = []
        self.bandwith_cloud = 100*1e6
        self.bandwith_edge = np.random.randint(100, 300, n_edge)*1e6
        # 信道增益和噪声的比值
        self.varphi = np.random.uniform(1, 10, n_edge)
        self.varphi_cloud = 3*1e-5

        # 精度需求
        self.accuracy_max = np.random.uniform(0.1, 0.6)

        # self.reset()


    # 获得用户最大发射功率
    def get_transmit_power(self):
        max_transmit_power = np.random.randint(1, 50)*1e-3
        return max_transmit_power

    # 获得针对云端的用户最大发射功率
    def get_cloud_transmit_power(self):
        cloud_transmit_power = 80*1e-3
        return cloud_transmit_power



