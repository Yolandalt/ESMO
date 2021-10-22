import numpy as np
import matplotlib.pyplot as plt
from sko.GA_TR import GA_EdgeVideo,GA_EdgeVideo_Y
import time
from examples.TargetRecognition.TargetRecognition import TR_Class

start = time.time()
np.random.seed(1)
n_edge = 5  # 边缘节点的数量
n_user = 8  # 用户数量
n_CNNmodel = 7  # CNN模型数量

# 创建自定义GA类
TR = TR_Class(n_edge = n_edge, n_user = n_user, n_CNNmodel = n_CNNmodel)
# 获得子染色体帧，以及染色体的长度（即基因数）
lengthEncode = TR.getEncodedLength()
# 获得Y染色体的基因数
lengthEncode_Y = TR.getEncodedLength_Y()
# 初始化模型缓存决策变量  n_edge*n_CNNModel矩阵（值为0或1）
caching_decision_all = TR.cache.init_caching_decision()
fitness_X = []
fitness_Y = []
last_offloading_decision = []
last_caching_decision = []
Max_iteration = 10
size_pop_X = 50
size_pop_Y = size_pop_X
max_iter_X = 180
max_iter_Y = max_iter_X


for i in range(Max_iteration):
    # 计算适度值
    def cal_total_cost_X(chromosome):
        '''
        目标函数. input：染色体, return 总的代价
        '''
        # 解码得到卸载决策user*frame
        fitnessValue = 0
        decoded_offloding_decison = TR.decodedChromosome(chromosome)
        # users_in_edges = TR.get_users_in_edge(decoded_offloding_decison)
        # a = ga_customize.get_cloud_fitness_value()
        fitnessValue = fitnessValue + TR.get_one_chromosome_fitness(decoded_offloding_decison,caching_decision_all)
        return fitnessValue

    # %% do 计算卸载GA
    ga_X = GA_EdgeVideo(func=cal_total_cost_X, caching_decision= caching_decision_all, last_best_offloading_decision=last_offloading_decision,
                        n_dim=lengthEncode, size_pop=size_pop_X, max_iter=max_iter_X, prob_mut=0.8)
    best_x,_ = ga_X.run()
    print("best_x:")
    print(best_x)
    last_offloading_decision = best_x  # 记录这轮最优的计算卸载决策，用于下一轮的迭代
    decoded_offloading_decision = TR.decodedChromosome(best_x)
    fitness_X = fitness_X+ga_X.generation_best_Y

    def cal_total_cost_Y(chromosome):
        '''
        目标函数. input：染色体, return 总的代价
        '''
        # 解码得到卸载决策user*frame
        fitnessValue = 0
        decoded_caching_decision_Y = TR.decodedChromosome_Y(chromosome)
        # users_in_edges = TR.get_users_in_edge(decoded_offloading_decision)
        # fitnessValue = fitnessValue + TR.get_one_chromosome_fitness(decoded_offloading_decision, decoded_caching_decision_Y)
        # 求模型缓存的最大精度
        fitnessValue_accuracy = TR.get_accuracy_fitnessValue(decoded_offloading_decision, decoded_caching_decision_Y)
        # 返回负值，因为要最大化
        return -fitnessValue_accuracy
        # return fitnessValue

    # 模型缓存
    ga_Y = GA_EdgeVideo_Y(func=cal_total_cost_Y, best_caching_decision= last_caching_decision, n_edge=n_edge,
                          n_user= n_user, n_CNNmodel=n_CNNmodel,
                          n_dim=lengthEncode, size_pop=size_pop_Y, max_iter=max_iter_Y, prob_mut=1)
    print("开始模型缓存决策优化")
    best_Y, _ = ga_Y.run()
    fitness_Y = fitness_Y+ga_Y.generation_best_Y
    caching_decision_all = TR.decodedChromosome_Y(best_Y)
    last_caching_decision = best_Y

    print(fitness_Y)
    print(len(fitness_Y))

end = time.time()
print('耗时: {}s'.format(end - start))
# %% plot
# plt.plot(ga_X.generation_best_Y)
# 写入文件
t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
dir = "./dataResults/fitnessValues_results_TR"+t+".txt"
f = open(dir, "w")
for item in fitness_X:
    f.write(str(round(item, 2))+ ",")
f.write('\n')
for item_Y in fitness_Y:
    f.write(str(round(item_Y, 2))+ ",")
f.close()
# 画图
# print(fitness_X)
fig, ax = plt.subplots(1, 2)
ax[0].plot(fitness_X)
ax[1].plot(fitness_Y)
# dir1 =
plt.savefig("./figure_plot/result_TR_.jpg")
plt.show()
