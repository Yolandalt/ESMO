import numpy as np
import math
import time
from EdgeVideoClass.Caching import ModelCaching
from EdgeVideoClass.Computation import ComputationOffload
from EdgeVideoClass.User import User_class
import random

class GA_customize:
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
        # 得到染色体编码长度
        self.lengthEncode_one_frame = self.__n_edge+2
        # 初始染色体数
        self.populationSize = 10

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

    # 初始化卸载决策变量X种群编码
    def getIntialPopulation(self, num_sub_chromosome, populationSize):
        chromosomes = np.zeros((populationSize, num_sub_chromosome*self.lengthEncode_one_frame), dtype=np.uint8)
        for i in range(populationSize):
            for j in range(num_sub_chromosome):
                # 随机选择一个frame决策内的一个数，设定为1，代表选中某个节点
                decision = np.random.randint(self.lengthEncode_one_frame*j,self.lengthEncode_one_frame*(j+1))
                chromosomes[i][decision] = 1
        return chromosomes

    # 初始化卸载决策变量X种群编码，需要满足资源限制
    def getIntialPopulation_Y(self, lengthEncode_Y, populationSize):
        # lengthEncode_Y = self.__n_edge*self.__n_CNNmodel
        chromosomes_Y = np.zeros((populationSize, lengthEncode_Y), dtype=np.uint8)
        for i in range(populationSize):
            for edge in range(self.__n_edge):  # 针对每一个边缘节点
                # 判断节点上的缓存资源是否足够
                # while self.cache.is_cache_full(edge):
                    # 足够，则随机选择一个CNN模型进行缓存
                    # self.caching_decision_all[i][np.random.randint(0, self.n_CNNModel)] = 1
                # 为该边缘节点随机选择缓存的模型数
                num_model = np.random.randint(0,self.__n_CNNmodel)
                decision = np.random.randint(self.__n_CNNmodel*edge,self.__n_CNNmodel*(edge+1),num_model)
                for item in decision:
                    chromosomes_Y[i][item] = 1
        return chromosomes_Y

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

    # 求出最优边缘计算能力表达式的分母
    def computing_denominator_f(self,users_in_an_edge_one):
        # 初始化
        denominator = 0
        for user in users_in_an_edge_one:  # user是对象
            for frame in range(user.n_frames):
                task_size = user.task_size[frame]
                CPU_clock = user.CPU_clock_per_bit[0]
                denominator = denominator + np.sqrt( task_size*CPU_clock)
        return denominator

    # 求出最优边缘计算能力
    def computing_optimal_f(self, user, frame, edge,users_in_edge_one):
        # 一趟染色体上一个边缘节点上的用户
        f_optimal_one = self.computation.freq_mec[0][edge] * np.sqrt(self.Users[user].task_size[frame] * self.Users[
                                                       user].CPU_clock_per_bit)[0]/self.computing_denominator_f(users_in_edge_one)
        return f_optimal_one

    # 计算本地计算的适应度
    def get_local_fitness_value(self, user, frame):
        user_CPU_clock_per_bit = 1500
        # print(self.Users[user].task_size[frame])
        fitness_values_for_local = self.Users[user].task_size[frame]* user_CPU_clock_per_bit*\
                                                     (1/self.Users[user].user_capability+
                                                      self.belta*np.square(self.Users[user].user_capability)*1e6*self.dr)
        # print(self.Users[user].user_capability)
        # print("fitness_values_for + local" +" "+ str(fitness_values_for_local))
        return fitness_values_for_local

    # 计算远程云的适应度
    def get_cloud_fitness_value(self, user, frame):
        CPU_clock = self.Users[user].CPU_clock_per_bit[0]
        cloud_f = self.computation.get_cloud_computing_capacities()
        task_size = self.Users[user].task_size[frame]
        power = self.Users[user].get_cloud_transmit_power()
        bandwith = self.Users[user].bandwith_cloud
        varphi = self.Users[user].varphi_cloud
        local_capacities = self.Users[user].user_capability
        a = (1+self.belta * power)/(bandwith* math.log2(1+varphi *power))
        b = CPU_clock* (1/cloud_f)
        # c = CPU_clock * self.belta*(np.square(local_capacities)*self.dr)
        fitness_values_for_cloud = task_size*((1+self.belta * power)/(bandwith* math.log2(1+varphi *power))+CPU_clock* (1/cloud_f) )
        # print("fitness_values_for_cloud" + str(fitness_values_for_cloud))
        return fitness_values_for_cloud

    # 计算边缘节点的适应度
    def get_edge_fitness_value(self, user, frame, edge, users_in_edges_one):
        # 获得edge的最优计算能力
        optimal_f = self.computing_optimal_f(user, frame, edge, users_in_edges_one[edge])
        task_size = self.Users[user].task_size[frame]
        power = self.Users[user].get_transmit_power()
        bandwidth = self.Users[user].bandwith_edge
        varphi = self.Users[user].varphi[edge]
        CPU_clock = self.Users[user].CPU_clock_per_bit[0]
        f_user = self.Users[user].user_capability
        a = (1 + self.belta * (power/2))/(bandwidth[edge] *math.log(1 + varphi * CPU_clock))
        b = CPU_clock*(1/optimal_f)
        # c = self.belta*CPU_clock*(np.square(f_user)*self.dr)
        fitness_values_for_edge = task_size * ((1 + self.belta * (power/2))/(bandwidth[edge] *math.log(1 + varphi * CPU_clock))+
                                               CPU_clock*(1/optimal_f))
        # print("fitness_values_for_edge" + str(fitness_values_for_edge))
        return fitness_values_for_edge

    def get_one_chromosome_fitness(self,decoded_offloding_decison,users_in_edges_one,decoded_caching_decision):
        # users_in_edges_one 一个染色体上所有节点的用户
        fitness_value_one = 0
        offloading_decision = decoded_offloding_decison
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                if offloading_decision[user][frame] == 0:  # 本地
                    # 为本地处理时延和本地能耗综合
                    fitness_value_one += self.get_local_fitness_value(user, frame)
                    # print("进行本地计算")
                elif offloading_decision[user][frame] == self.__n_edge + 1:  # 远程云
                    # print("进行云计算")
                    fitness_value_one += self.get_cloud_fitness_value(user, frame)
                else:  # 卸载到边缘节点edge上
                    # print("进行边缘计算")
                    # 需要判断是否存在满足精确度的CNN模型
                    # 先找出缓存在该节点上的CNN模型(找出元素为1的所有下标)
                    edge = int(offloading_decision[user][frame])-1
                    a = decoded_caching_decision[edge]
                    CNN_list = [i for i in range(len(a)) if a[i] == 1]

                    if CNN_list:
                        # 取出对应精度
                        accuray_caching = []
                        for i in range(len(CNN_list)):
                            accuray_caching.append(self.cache.accuracy_provide[i])
                        # 取最大值
                        max_accuracy_CNN = max(accuray_caching)
                        # 和用户精度需求进行判断
                        if self.Users[user].accuracy_max > max_accuracy_CNN:
                            # CNN模型不能满足用户精度
                            # 卸载到云
                            # print("CNN模型不能满足用户精度")
                            fitness_value_one += self.get_cloud_fitness_value(user, frame)
                        else:
                            # 可以满足精度 （在边缘节点进行计算）
                            # print("可以满足精度 （在边缘节点进行计算）")
                            fitness_value_one += self.get_edge_fitness_value(user, frame, edge, users_in_edges_one)
                    else:
                        # print("该节点上没有缓存CNN模型")
                        # 卸载到云
                        fitness_value_one += self.get_cloud_fitness_value(user, frame)

        return fitness_value_one

    # 计算所有路线的适应度
    def get_all_chromosome_fitness_value(self, populationSize, decoded_offloading_decision, users_in_edge_,decoded_caching_decision_all):
        fitness_values = np.zeros(populationSize)
        for i in range(populationSize):
            f_value = self.get_one_chromosome_fitness(i,decoded_offloading_decision,users_in_edge_[i],decoded_caching_decision_all[i])
            fitness_values[i] = f_value
        return fitness_values

    # 计算模型缓存的适应度，即在满足用户精度需求之后所有用户的平均精度最大化
    def get_accuracy_fitnessValue(self, decoded_offloading_decision, decoded_model_caching_decision):
        accuracy_fitnessValue = 0
        for user in range(len(decoded_offloading_decision)):
            for frame in range(len(decoded_offloading_decision[user])):
                if decoded_offloading_decision[user][frame] == 0:  # 本地
                    # 取模型的最低精度
                    accuracy_fitnessValue += np.min(self.cache.accuracy_provide)
                elif decoded_offloading_decision[user][frame] == self.__n_edge + 1:  # 远程云
                    accuracy_fitnessValue += np.max(self.cache.accuracy_provide)
                else:  # 卸载到边缘节点edge上
                    # 需要判断是否存在满足精确度的CNN模型
                    # 先找出缓存在该节点上的CNN模型(找出元素为1的所有下标)
                    edge = int(decoded_offloading_decision[user][frame])-1
                    a = decoded_model_caching_decision[edge]
                    CNN_list = [i for i in range(len(a)) if a[i] == 1]

                    if CNN_list:
                        # 取出对应精度
                        accuracy_caching = []
                        for i in range(len(CNN_list)):
                            accuracy_caching.append(self.cache.accuracy_provide[i])
                        # 取最大值
                        max_accuracy_CNN = max(accuracy_caching)
                        # 和用户精度需求进行判断
                        if self.Users[user].accuracy_max > max_accuracy_CNN:
                            # CNN模型不能满足用户精度
                            # 卸载到云
                            # print("CNN模型不能满足用户精度")
                            accuracy_fitnessValue += np.max(self.cache.accuracy_provide)
                        else:
                            # 可以满足精度 （在边缘节点进行计算）
                            # print("可以满足精度 （在边缘节点进行计算）")
                            accuracy_fitnessValue += max_accuracy_CNN
                    else:
                        # print("该节点上没有缓存CNN模型")
                        # 卸载到云
                        accuracy_fitnessValue += np.max(self.cache.accuracy_provide)
        return accuracy_fitnessValue

    # 新种群选择
    def selectNewPopulation(self,chromosomes, fitness_values):
        m, n = chromosomes.shape
        newpopulation =np.zeros((m, n), dtype=np.uint8)
        # 适度值越小越应该被选择
        # max_value = max(fitness_values)
        # fitness_values_reverse = fitness_values.copy()
        # fitness_values_reverse = [max_value + 1 - x for x in fitness_values]
        # print(fitness_values)
        # probability = fitness_values_reverse / np.sum(fitness_values_reverse)
        probability = fitness_values / np.sum(fitness_values)
        for i in range(m):
            choice = np.random.choice(range(m), p=probability)  # 倾向于选择适度值小的
            newpopulation[i] = chromosomes[choice]
        return newpopulation

    # 新种群交叉操作
    def crossover(self, newpopulation, Pc=0.8):
        """
        :param newpopulation: 新种群
        :param Pc: 交叉概率默认是0.8
        :return: 交叉后得到的新种群
        """
        # 根据交叉概率计算需要进行交叉的个体个数
        m, n = newpopulation.shape
        numbers = np.uint8(m * Pc)
        # 确保进行交叉的染色体个数是偶数个
        if numbers % 2 != 0:
            numbers += 1
        # 交叉后得到的新种群
        updatepopulation = np.zeros((m, n), dtype=np.uint8)
        # 产生随机索引
        index = random.sample(range(m), numbers)
        # 不进行交叉的染色体进行复制
        for i in range(m):
            if not index.__contains__(i):
                updatepopulation[i, :] = newpopulation[i, :]
        # crossover
        while len(index) > 0:
            a = index.pop()  # 取最后一位
            b = index.pop()
            # 随机产生一个交叉点
            # 必须产生在每个决策分界点
            num_frames,lengthEncode = self.getEncodedLength()
            crossoverPoint_frame = random.sample(range(1, num_frames), 1)
            crossoverPoint_frame = crossoverPoint_frame[0]
            crossoverPoint = crossoverPoint_frame*self.lengthEncode_one_frame
            # one-single-point crossover
            updatepopulation[a, 0:crossoverPoint] = newpopulation[a, 0:crossoverPoint]
            updatepopulation[a, crossoverPoint:] = newpopulation[b, crossoverPoint:]
            updatepopulation[b, 0:crossoverPoint] = newpopulation[b, 0:crossoverPoint]
            updatepopulation[b, crossoverPoint:] = newpopulation[a, crossoverPoint:]
        return updatepopulation

    # 新种群Y的交叉操作
    def crossover_Y(self, newpopulation_Y, Pc=0.8):
        """
        :param newpopulation_Y: 新种群
        :param Pc: 交叉概率默认是0.8
        :return: 交叉后得到的新种群
        """
        # 根据交叉概率计算需要进行交叉的个体个数
        m, n = newpopulation_Y.shape
        numbers = np.uint8(m * Pc)
        # 确保进行交叉的染色体个数是偶数个
        if numbers % 2 != 0:
            numbers += 1
        # 交叉后得到的新种群
        updatepopulation_Y = np.zeros((m, n), dtype=np.uint8)
        # 产生随机索引
        index_Y = random.sample(range(m), numbers)
        # 不进行交叉的染色体进行复制
        for i in range(m):
            if not index_Y.__contains__(i):
                updatepopulation_Y[i, :] = newpopulation_Y[i, :]
        # crossover
        while len(index_Y) > 0:
            a = index_Y.pop()  # 取最后一位
            b = index_Y.pop()
            # 随机产生一个交叉点
            # 必须产生在每个边缘节点edge缓存决策分界点
            lengthEncode_Y = self.getEncodedLength_Y()
            crossoverPoint_edge = random.sample(range(1, lengthEncode_Y), 1)
            crossoverPoint_edge = crossoverPoint_edge[0]
            crossoverPoint = crossoverPoint_edge * self.__n_CNNmodel
            # one-single-point crossover
            updatepopulation_Y[a, 0:crossoverPoint] = newpopulation_Y[a, 0:crossoverPoint]
            updatepopulation_Y[a, crossoverPoint:] = newpopulation_Y[b, crossoverPoint:]
            updatepopulation_Y[b, 0:crossoverPoint] = newpopulation_Y[b, 0:crossoverPoint]
            updatepopulation_Y[b, crossoverPoint:] = newpopulation_Y[a, crossoverPoint:]
        return updatepopulation_Y

    # 染色体变异
    def mutation(self, population, Pm=0.1):
        """
        :param population: 经交叉后得到的种群
        :param Pm: 变异概率默认是0.01
        :return: 经变异操作后的新种群
        """
        updatepopulation = np.copy(population)
        m, n = population.shape
        # 计算需要变异的基因个数(也可以是frame的个数)
        mutation_num = np.uint8(m * n * Pm)
        # 每个染色体的frame数
        frame_num = int(n/self.lengthEncode_one_frame)
        # 随机抽取mutation_num个基因进行基本位变异
        mutationFrameIndex = random.sample(range(0, m * frame_num), mutation_num)
        # 确定每个将要变异的frame在整个染色体中的基因座(即基因/frame的具体位置)
        for gene_frame in mutationFrameIndex:
            # 确定变异基因位于第几个染色体
            # print("gene_frame" + str(gene_frame))
            chromosomeIndex = gene_frame // frame_num  # 整除
            if gene_frame >= frame_num:
                remainder = gene_frame % frame_num # 取余数
                # 如果余数大于0,则应该在下一条染色体中
                if remainder > 0:
                    chromosomeIndex = chromosomeIndex + 1-1  # 单数因为下标从0开始
            else:
                remainder = gene_frame  # 直接在第0个染色体内
            # 确定变异基因位于当前染色体的第几个frame位， 即为余数
            frame_index = remainder
            # print("frame_index" + str(frame_index))
            # 确定该frame的起始bit的index位置
            bit_start = (frame_index)*self.lengthEncode_one_frame
            bit_end = (frame_index+1)*self.lengthEncode_one_frame-1
            # 取出变异的frame
            muta_frame = population[chromosomeIndex][bit_start:bit_end+1]
            # 因为所有节点只能卸载到一个计算节点，得保证这个frame里只有一个1
            index_1 = self.get_index1(muta_frame,1)
            index_0 = self.get_index1(muta_frame,0)
            # 将之前为1的bit位改为0

            updatepopulation[chromosomeIndex][bit_start+index_1[0]] = 0
            # 随机确定该基因在frame里的位置
            geneIndex_frame = random.sample(index_0, 1)[0]
            updatepopulation[chromosomeIndex][bit_start+geneIndex_frame] = 1

        return updatepopulation

    # Y染色体变异
    def mutation_Y(self, population_Y, Pm=0.1):
        """
        :param population_Y: 经交叉后得到的种群
        :param Pm: 变异概率默认是0.01
        :return: 经变异操作后的新种群
        """
        updatepopulation_Y = np.copy(population_Y)
        m, n = population_Y.shape
        # 计算需要变异的基因个数
        gene_num = np.uint8(m * n * Pm)
        # 将所有的基因按照序号进行10进制编码，则共有m*n个基因
        # 随机抽取gene_num个基因进行基本位变异
        mutationGeneIndex = random.sample(range(0, m * n), gene_num)
        # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)
        for gene in mutationGeneIndex:
            # 确定变异基因位于第几个染色体
            chromosomeIndex = gene // n
            # 确定变异基因位于当前染色体的第几个基因位
            geneIndex = gene % n
            # mutation
            if updatepopulation_Y[chromosomeIndex, geneIndex] == 0:
                updatepopulation_Y[chromosomeIndex, geneIndex] = 1
            else:
                updatepopulation_Y[chromosomeIndex, geneIndex] = 0
        return updatepopulation_Y

    # 返回列表中某个元素的所有下标
    def get_index1(self, lst=None, item=0):
        return [index for (index, value) in enumerate(lst) if value == item]