import numpy as np
import math
import time
from EdgeVideoClass.Caching import ModelCaching
from EdgeVideoClass.Computation import ComputationOffload
from EdgeVideoClass.User import User_class
import random
import json

class TR_Class:
    def __init__(self,
        n_edge=6,  # 边缘节点的数量
        n_user = 5,  # 用户数量
        n_CNNmodel = 8,  # CNN模型数量
                 ):
        self.__n_edge = n_edge
        self.__n_user = n_user
        self.__n_CNNmodel = n_CNNmodel
        # 定义类的对象
        self.cache = ModelCaching(n_edge=self.__n_edge, n_CNNModel=self.__n_CNNmodel)
        self.computation = ComputationOffload(n_edge=self.__n_edge, n_user=self.__n_user)
        # 创建用户
        self.Users = []
        for i in range(self.__n_user):
            self.Users.append(User_class(task_arrival_rate=0.5, energy_unit=5e-5, n_edge=self.__n_edge))

        # 初始化模型缓存决策变量  n_edge*n_CNNModel矩阵（值为0或1）
        self.caching_decision_all = self.cache.init_caching_decision()

        # 权值
        self.belta = 0.5
        # 电容系数
        self.dr = 1e-28
        # 有不同CNN模型
        # 云
        self.TR_Result_cloud = []
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_C4_1x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_C4_3x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_DC5_3x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_FPN_1x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_C4_3x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_DC5_3x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_FPN_3x.json"))
        # 本地
        self.TR_Result_local = self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_50_C4_1x.json")
        # self.TR_Result_local = self.ReadTRResults("./TargetRecognition/result_local.json")
        # self.TR_Result_edge = [ [] for j in range(2) ]
        # for edge in range(self.__n_edge):
        #     dir_edge = "./TargetRecognition/result_edge"+str(edge)+".json"
        #     self.TR_Result_edge.append(self.ReadTRResults(dir_edge))

        self.TR_Result_edge_all = []
        # node1
        self.TR_Result_node1 = []
        self.TR_Result_node1.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_50_C4_1x.json"))
        self.TR_Result_node1.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_50_C4_3x.json"))
        self.TR_Result_node1.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_50_DC5_3x.json"))
        self.TR_Result_node1.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_50_FPN_1x.json"))
        self.TR_Result_node1.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_101_C4_3x.json"))
        self.TR_Result_node1.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_101_DC5_3x.json"))
        self.TR_Result_node1.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_101_FPN_3x.json"))
        self.TR_Result_edge_all.append(self.TR_Result_node1)

        # node2
        self.TR_Result_node2 = []
        self.TR_Result_node2.append(self.ReadTRResults("./TargetRecognition/data/local/mask_rcnn_R_50_C4_1x.json"))
        self.TR_Result_node2.append(self.ReadTRResults("./TargetRecognition/data/local/mask_rcnn_R_50_C4_3x.json"))
        self.TR_Result_node2.append(self.ReadTRResults("./TargetRecognition/data/local/mask_rcnn_R_50_DC5_3x.json"))
        self.TR_Result_node2.append(self.ReadTRResults("./TargetRecognition/data/local/mask_rcnn_R_50_FPN_1x.json"))
        self.TR_Result_node2.append(self.ReadTRResults("./TargetRecognition/data/local/mask_rcnn_R_101_C4_3x.json"))
        self.TR_Result_node2.append(self.ReadTRResults("./TargetRecognition/data/local/mask_rcnn_R_101_DC5_3x.json"))
        self.TR_Result_node2.append(self.ReadTRResults("./TargetRecognition/data/local/mask_rcnn_R_101_FPN_3x.json"))
        self.TR_Result_edge_all.append(self.TR_Result_node2)

        # node3
        self.TR_Result_node3 = []
        self.TR_Result_node3.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_C4_1x.json"))
        self.TR_Result_node3.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_C4_3x.json"))
        self.TR_Result_node3.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_DC5_3x.json"))
        self.TR_Result_node3.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_FPN_1x.json"))
        self.TR_Result_node3.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_C4_3x.json"))
        self.TR_Result_node3.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_DC5_3x.json"))
        self.TR_Result_node3.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_FPN_3x.json"))
        self.TR_Result_edge_all.append(self.TR_Result_node3)

        # node4
        self.TR_Result_node4 = []
        self.TR_Result_node4.append(self.ReadTRResults("./TargetRecognition/data/242/mask_rcnn_R_50_C4_1x.json"))
        self.TR_Result_node4.append(self.ReadTRResults("./TargetRecognition/data/242/mask_rcnn_R_50_C4_3x.json"))
        self.TR_Result_node4.append(self.ReadTRResults("./TargetRecognition/data/242/mask_rcnn_R_50_DC5_3x.json"))
        self.TR_Result_node4.append(self.ReadTRResults("./TargetRecognition/data/242/mask_rcnn_R_50_FPN_1x.json"))
        self.TR_Result_node4.append(self.ReadTRResults("./TargetRecognition/data/242/mask_rcnn_R_101_C4_3x.json"))
        self.TR_Result_node4.append(self.ReadTRResults("./TargetRecognition/data/242/mask_rcnn_R_101_DC5_3x.json"))
        self.TR_Result_node4.append(self.ReadTRResults("./TargetRecognition/data/242/mask_rcnn_R_101_FPN_3x.json"))
        self.TR_Result_edge_all.append(self.TR_Result_node4)

        # node5
        self.TR_Result_node5 = []
        self.TR_Result_node5.append(self.ReadTRResults("./TargetRecognition/data/jf/mask_rcnn_R_50_C4_1x.json"))
        self.TR_Result_node5.append(self.ReadTRResults("./TargetRecognition/data/jf/mask_rcnn_R_50_C4_3x.json"))
        self.TR_Result_node5.append(self.ReadTRResults("./TargetRecognition/data/jf/mask_rcnn_R_50_DC5_3x.json"))
        self.TR_Result_node5.append(self.ReadTRResults("./TargetRecognition/data/jf/mask_rcnn_R_50_FPN_1x.json"))
        self.TR_Result_node5.append(self.ReadTRResults("./TargetRecognition/data/jf/mask_rcnn_R_101_C4_3x.json"))
        self.TR_Result_node5.append(self.ReadTRResults("./TargetRecognition/data/jf/mask_rcnn_R_101_DC5_3x.json"))
        self.TR_Result_node5.append(self.ReadTRResults("./TargetRecognition/data/jf/mask_rcnn_R_101_FPN_3x.json"))
        self.TR_Result_edge_all.append(self.TR_Result_node5)

        # node6
        self.TR_Result_node6 = []
        self.TR_Result_node6.append(self.ReadTRResults("./TargetRecognition/data/jf/mask_rcnn_R_50_C4_1x.json"))
        self.TR_Result_node6.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_50_C4_3x.json"))
        self.TR_Result_node6.append(self.ReadTRResults("./TargetRecognition/data/local/mask_rcnn_R_50_DC5_3x.json"))
        self.TR_Result_node6.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_FPN_1x.json"))
        self.TR_Result_node6.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_101_C4_3x.json"))
        self.TR_Result_node6.append(self.ReadTRResults("./TargetRecognition/data/242/mask_rcnn_R_101_DC5_3x.json"))
        self.TR_Result_node6.append(self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_101_FPN_3x.json"))
        self.TR_Result_edge_all.append(self.TR_Result_node6)

        a = self.TR_Result_edge_all

    def ReadTRResults(self, dir):
        with open(dir, 'r') as load_f:
            load_dict = json.load(load_f)
        return load_dict

    # 获得染色体总长度（即所有用户和所有帧）
    def getEncodedLength(self):
        lengthEncode = 0  # 有多少个决策变量
        for user in range(self.__n_user):
            lengthEncode += self.Users[user].n_frames
        return lengthEncode

    # 确定缓存决策变量Y的染色体长度
    def getEncodedLength_Y(self):
        # 每个边缘节点上有一个缓存决策变量，针对每个边缘节点，存模型的序列号
        lengthEncode_Y = 0
        # 决策变量的总和
        for edge in range(self.__n_edge):
            lengthEncode_Y += self.cache.num_CNN_in_edge[edge]
        return lengthEncode_Y

    # 染色体解码得到表现型的解
    # 这里的表现型为卸载决策变量即n_user*用户对应的frame，对应值为计算节点的标号
    def decodedChromosome(self, chromosome):
        # 初始化卸载决策n_user*n_frame,针对每个用户的每个frame，0为本地；1-6为边缘节点；7为远程云
        offloading_decision = []
        # 开始解码
        # 取一个染色体
        start_index = 0
        for user in range(self.__n_user):
            end_index = start_index + self.Users[user].n_frames
            # 取某个用户的染色体片段
            chromosome_slice = chromosome[start_index:end_index]
            offloading_decision.append(chromosome_slice)
            start_index += self.Users[user].n_frames
        return offloading_decision

    # 染色体解码得到表现型的解
    # 这里的表现型为缓存决策变量即populationSize*n_edge*对应的CNN模型数量
    def decodedChromosome_Y(self, chromosome_Y):

        # 初始化缓存决策n_edge*n_CNNmodel,针对每个边缘节点上的CNN模型，0为缓存，1为没有缓存
        decoded_caching_decision = []
        # 开始解码
        # 针对每一个边缘节点
        for edge in range(self.__n_edge):
            decoded_caching_decision.append(chromosome_Y[edge*self.__n_CNNmodel:(edge+1)*self.__n_CNNmodel])
        return decoded_caching_decision

    def decodedChromosome_both(self,chromosome):
        # chromosome前面是模型缓存决策，后面是计算卸载决策
        lengthEncode_Caching = self.getEncodedLength_Y()
        lengthEncode_offloading = self.getEncodedLength()
        chromosome_Y = chromosome[:lengthEncode_Caching]
        chromosome_X = chromosome[lengthEncode_Caching:]
        # 模型缓存
        decoded_caching_decision = []*self.__n_edge
        for edge in range(self.__n_edge):
            decoded_caching_decision.append(np.zeros(self.__n_CNNmodel))
        # 开始解码
        # 针对每一个边缘节点
        last_length_model = 0
        last_edge_num_CNN = 0
        for edge in range(self.__n_edge):
            last_edge_num_CNN = self.cache.num_CNN_in_edge[edge]
            for model in range(last_edge_num_CNN):
                index_model = last_length_model+int(chromosome_Y[model])
                decoded_caching_decision[edge][index_model] = 1
        last_length_model += last_edge_num_CNN
        # 计算卸载
        decoded_offloading_decision = []
        # 开始解码
        # 取一个染色体
        start_index = 0
        for user in range(self.__n_user):
            end_index = start_index + self.Users[user].n_frames
            # 取某个用户的染色体片段
            chromosome_slice = chromosome[start_index:end_index]
            decoded_offloading_decision.append(chromosome_slice)
            start_index += self.Users[user].n_frames
        return decoded_caching_decision, decoded_offloading_decision

    # 求出一个边缘节点上缓存了多少用户（用户数）.主要求总的frame数
    # 其实应该求边缘节点上的卸载frame！！！！！记得优化
    def get_users_in_edge(self, offloading_decision):
        # 初始化 n_edge
        users_in_edges = [ [] for j in range(self.__n_edge) ]
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                for edge in range(self.__n_edge):
                    # print(offloading_decision)
                    if offloading_decision[user][frame] == edge+1:
                        users_in_edges[edge].append(self.Users[user])
        return users_in_edges

    # 计算本地计算的适应度
    def get_local_fitness_value(self, frame_id):
        # fitness_values_for_local = 0
        # 取出精度
        pic_precision = self.TR_Result_local[frame_id].get("pic_precision")
        # 取出时延
        delay = self.TR_Result_local[frame_id].get("time_used")
        # 取出能耗
        energy_cost = self.TR_Result_local[frame_id].get("cpu_time_used")
        # energy_addition = np.random.uniform(50, 80)
        energy_addition = np.random.uniform(20, 25)
        # 适度值计算
        fitness_values_for_local = delay + self.belta*energy_cost+energy_addition

        # print("fitness_values_for + local" +" "+ str(fitness_values_for_local))
        return fitness_values_for_local,pic_precision

    # 计算远程云的适应度
    def get_cloud_fitness_value(self, frame_id):
        # fitness_values_for_cloud = 0
        pic_precision_all_model = []
        for model in range(len(self.TR_Result_cloud)):
            # 取出精度
            pic_precision_all_model.append(self.TR_Result_cloud[model][frame_id].get("pic_precision"))
        # 求精度最大值
        pic_precision_max = max(pic_precision_all_model)
        select_model = self.get_index1(pic_precision_all_model,pic_precision_max)[0]
        # 取出时延
        delay = self.TR_Result_cloud[select_model][frame_id].get("time_used")
        # 取出能耗
        energy_cost = self.TR_Result_cloud[select_model][frame_id].get("cpu_time_used")
        # 加上传输时延
        # delay_addition = np.random.uniform(80,100)
        delay_addition = np.random.uniform(20, 40)
        # 适度值计算
        fitness_values_for_cloud = delay + self.belta * energy_cost+delay_addition
        # print("fitness_values_for_cloud" + str(fitness_values_for_cloud))
        return fitness_values_for_cloud, pic_precision_max

    # 计算边缘节点的适应度
    def get_edge_fitness_value(self, frame_id, edge_id,model_id):
        # 获得edge的最优计算能力
        # 取出精度
        pic_precision = self.TR_Result_edge_all[edge_id][model_id][frame_id].get("pic_precision")
        # 取出时延
        delay = self.TR_Result_edge_all[edge_id][model_id][frame_id].get("time_used")
        # 取出能耗
        energy_cost = self.TR_Result_edge_all[edge_id][model_id][frame_id].get("cpu_time_used")
        # 适度值计算
        fitness_values_for_edge = delay + self.belta * energy_cost
        if edge_id == 0:
            fitness_values_for_edge += np.random.randint(0,5)
        if edge_id == 1:
            fitness_values_for_edge += np.random.randint(22,30)
        if edge_id == 2:
            fitness_values_for_edge += np.random.randint(10,25)
        if edge_id == 3:
            fitness_values_for_edge += np.random.randint(15,20)
        if edge_id == 4:
            fitness_values_for_edge += np.random.randint(20,22)
        if edge_id == 5:
            fitness_values_for_edge += np.random.randint(20,25)

        # if edge_id == 0:
        #     fitness_values_for_edge += np.random.randint(0,5)
        # if edge_id == 1:
        #     fitness_values_for_edge += np.random.randint(22,30)
        # if edge_id == 2:
        #     fitness_values_for_edge += np.random.randint(10,15)
        # if edge_id == 3:
        #     fitness_values_for_edge += np.random.randint(15,20)
        # if edge_id == 4:
        #     fitness_values_for_edge += np.random.randint(20,22)
        # if edge_id == 5:
        #     fitness_values_for_edge += np.random.randint(20,25)
        # print("fitness_values_for_edge" + str(fitness_values_for_edge))
        return fitness_values_for_edge,pic_precision

    def get_one_chromosome_fitness(self, decoded_offloding_decison,decoded_caching_decision):
        # users_in_edges_one 一个染色体上所有节点的用户
        fitness_value_one = 0
        offloading_decision = decoded_offloding_decison
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                if offloading_decision[user][frame] == 0:  # 本地
                    # 为本地处理时延和本地能耗综合
                    one, _ = self.get_local_fitness_value(frame_id)
                    fitness_value_one += one
                    # print("进行本地计算")
                elif offloading_decision[user][frame] == self.__n_edge + 1:  # 远程云
                    # print("进行云计算")
                    one, _ = self.get_cloud_fitness_value(frame_id)
                    fitness_value_one += one
                else:  # 卸载到边缘节点edge上
                    # print("进行边缘计算")
                    # 需要判断是否存在满足精确度的CNN模型
                    # 先找出缓存在该节点上的CNN模型(找出元素为1的所有下标)
                    edge = int(offloading_decision[user][frame])-1
                    print(edge)
                    print(offloading_decision)
                    a = decoded_caching_decision[edge]
                    CNN_list = [i for i in range(len(a)) if a[i] == 1]

                    if CNN_list:
                        # 取出对应精度
                        accuray_caching = []
                        for i in range(len(CNN_list)):
                            accuray_model = self.TR_Result_edge_all[edge][CNN_list[i]][frame_id].get("pic_precision")
                            accuray_caching.append(accuray_model)
                        # 取最大值
                        max_accuracy_CNN = max(accuray_caching)
                        corresponding_model_index = self.get_index1(accuray_caching,max_accuracy_CNN)[0]
                        # 和用户精度需求进行判断
                        # if self.Users[user].accuracy_max > max_accuracy_CNN:
                        #     # CNN模型不能满足用户精度
                        #     # 卸载到云
                        #     # print("CNN模型不能满足用户精度")
                        #     one, _ = self.get_cloud_fitness_value(frame_id)
                        #     fitness_value_one += one
                        # else:
                        #     # 可以满足精度 （在边缘节点进行计算）
                        #     #     print("可以满足精度 （在边缘节点进行计算）")
                        #     model_index = CNN_list[corresponding_model_index]
                        #     one, _ = self.get_edge_fitness_value(frame_id,edge,model_index)
                        #     fitness_value_one += one
                        model_index = CNN_list[corresponding_model_index]
                        one, _ = self.get_edge_fitness_value(frame_id, edge, model_index)
                        fitness_value_one += one
                    else:
                        # print("该节点上没有缓存CNN模型")
                        # 卸载到云
                        one, _ = self.get_cloud_fitness_value(frame_id)
                        fitness_value_one += one

        return fitness_value_one

    def get_accuracy_fitnessValue(self, decoded_offloading_decision, decoded_model_caching_decision):
        accuracy_fitnessValue = 0
        for user in range(len(decoded_offloading_decision)):
            for frame in range(len(decoded_offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                if decoded_offloading_decision[user][frame] == 0:   # 本地
                    # 取模型的最低精度
                    _, one = self.get_local_fitness_value(frame_id)
                    accuracy_fitnessValue += one
                elif decoded_offloading_decision[user][frame] == self.__n_edge + 1:  # 远程云
                    _, one = self.get_cloud_fitness_value(frame_id)
                    accuracy_fitnessValue += one
                else:  # 卸载到边缘节点edge上
                    # 需要判断是否存在满足精确度的CNN模型
                    # 先找出缓存在该节点上的CNN模型(找出元素为1的所有下标)
                    edge = int(decoded_offloading_decision[user][frame])-1
                    a = decoded_model_caching_decision[edge]
                    CNN_list = [i for i in range(len(a)) if a[i] == 1]

                    if CNN_list:
                        # 取出对应精度
                        accuray_caching = []
                        for i in range(len(CNN_list)):
                            accuray_model = self.TR_Result_edge_all[edge][CNN_list[i]][frame_id].get("pic_precision") * 10
                            accuray_caching.append(accuray_model)
                        # 取最大值
                        max_accuracy_CNN = max(accuray_caching)
                        corresponding_model_index = self.get_index1(accuray_caching,max_accuracy_CNN)[0]
                        # 和用户精度需求进行判断
                        # if self.Users[user].accuracy_max > max_accuracy_CNN:
                        #     # CNN模型不能满足用户精度
                        #     # 卸载到云
                        #     # print("CNN模型不能满足用户精度")
                        #     _, one = self.get_cloud_fitness_value(frame_id)
                        #     accuracy_fitnessValue -= one
                        # else:
                        #     # 可以满足精度 （在边缘节点进行计算）
                        #     # print("可以满足精度 （在边缘节点进行计算）")
                        #     accuracy_fitnessValue += max_accuracy_CNN
                        #     _, one = self.get_edge_fitness_value(frame_id, edge, corresponding_model_index)
                        #     accuracy_fitnessValue += one
                        _, one = self.get_edge_fitness_value(frame_id,edge,corresponding_model_index)
                        accuracy_fitnessValue += one
                    else:
                        # print("该节点上没有缓存CNN模型")
                        # 卸载到云
                        _, one = self.get_cloud_fitness_value(frame_id)
                        accuracy_fitnessValue -= one
                        # accuracy_fitnessValue += np.max(self.cache.accuracy_provide)
        return accuracy_fitnessValue

    def get_one_chromosome_fitness_local(self, decoded_offloding_decison,decoded_caching_decision):
        # users_in_edges_one 一个染色体上所有节点的用户
        fitness_value_one = 0
        offloading_decision = decoded_offloding_decison
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                # 只选本地
                # 为本地处理时延和本地能耗综合
                one, _ = self.get_local_fitness_value(frame_id)
                fitness_value_one += one
        return fitness_value_one

    def get_accuracy_fitnessValue_local(self, decoded_offloading_decision, decoded_model_caching_decision):
        accuracy_fitnessValue = 0
        for user in range(len(decoded_offloading_decision)):
            for frame in range(len(decoded_offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                # 本地
                # 取模型的最低精度
                _, one = self.get_local_fitness_value(frame_id)
                accuracy_fitnessValue += one
        return accuracy_fitnessValue

    def get_one_chromosome_fitness_cloud(self, decoded_offloding_decison, decoded_caching_decision):
        # users_in_edges_one 一个染色体上所有节点的用户
        fitness_value_one = 0
        offloading_decision = decoded_offloding_decison
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                # 只选本地
                # 为本地处理时延和本地能耗综合
                one, _ = self.get_cloud_fitness_value(frame_id)
                fitness_value_one += one
        return fitness_value_one

    def get_accuracy_fitnessValue_cloud(self, decoded_offloading_decision, decoded_model_caching_decision):
        accuracy_fitnessValue = 0
        for user in range(len(decoded_offloading_decision)):
            for frame in range(len(decoded_offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                # 本地
                # 取模型的最低精度
                _, one = self.get_cloud_fitness_value(frame_id)
                accuracy_fitnessValue += one
        return accuracy_fitnessValue

    # 返回列表中某个元素的所有下标
    def get_index1(self, lst=None, item=0):
        return [index for (index, value) in enumerate(lst) if value == item]