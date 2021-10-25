import numpy as np
import matplotlib.pyplot as plt
from sko.GA_TR import GA_EdgeVideo,GA_EdgeVideo_Y
import time
from examples.TargetRecognition.TargetRecognition import TR_Class

start = time.time()
np.random.seed(1)
n_edge = 5  # Number of edge nodes
n_user = 8  # amount of users
n_CNNmodel = 7  # Number of CNN models

# Create a custom GA class
TR = TR_Class(n_edge = n_edge, n_user = n_user, n_CNNmodel = n_CNNmodel)
# Get the child chromosome frame and the length of the chromosome (i.e., the number of genes)
lengthEncode = TR.getEncodedLength()
# Get the number of genes on the Y chromosome
lengthEncode_Y = TR.getEncodedLength_Y()
# Initialize the model cache decision variable n_edge*n_CNNModel matrix (value is 0 or 1)
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
    # Calculate the moderate value
    def cal_total_cost_X(chromosome):
        '''
        Objective function
        input：chromosome, return: the total cost
        '''
        # Decoding to get offloading decision user*frame
        fitnessValue = 0
        decoded_offloding_decison = TR.decodedChromosome(chromosome)
        fitnessValue = fitnessValue + TR.get_one_chromosome_fitness(decoded_offloding_decison,caching_decision_all)
        return fitnessValue

    # frame scheduling
    ga_X = GA_EdgeVideo(func=cal_total_cost_X, caching_decision= caching_decision_all, last_best_offloading_decision=last_offloading_decision,
                        n_dim=lengthEncode, size_pop=size_pop_X, max_iter=max_iter_X, prob_mut=0.8)
    best_x,_ = ga_X.run()
    last_offloading_decision = best_x   # Record this round of optimal computational offloading decisions for use in the next round of iterations
    decoded_offloading_decision = TR.decodedChromosome(best_x)
    fitness_X = fitness_X+ga_X.generation_best_Y

    def cal_total_cost_Y(chromosome):
        '''
        Objective function
        input：chromosome, return: the total cost
        '''
        fitnessValue = 0
        decoded_caching_decision_Y = TR.decodedChromosome_Y(chromosome)
        # Find the maximum accuracy of the model cache
        fitnessValue_accuracy = TR.get_accuracy_fitnessValue(decoded_offloading_decision, decoded_caching_decision_Y)
        # Returns a negative value because it wants to maximize
        return -fitnessValue_accuracy

    # Model cache
    ga_Y = GA_EdgeVideo_Y(func=cal_total_cost_Y, best_caching_decision= last_caching_decision, n_edge=n_edge,
                          n_user= n_user, n_CNNmodel=n_CNNmodel,
                          n_dim=lengthEncode, size_pop=size_pop_Y, max_iter=max_iter_Y, prob_mut=1)
    best_Y, _ = ga_Y.run()
    fitness_Y = fitness_Y+ga_Y.generation_best_Y
    caching_decision_all = TR.decodedChromosome_Y(best_Y)
    last_caching_decision = best_Y


end = time.time()
print('time consuming: {}s'.format(end - start))

t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
dir = "./dataResults/fitnessValues_results_TR"+t+".txt"
f = open(dir, "w")
for item in fitness_X:
    f.write(str(round(item, 2))+ ",")
f.write('\n')
for item_Y in fitness_Y:
    f.write(str(round(item_Y, 2))+ ",")
f.close()

fig, ax = plt.subplots(1, 2)
ax[0].plot(fitness_X)
ax[1].plot(fitness_Y)

plt.savefig("./figure_plot/result_TR_.jpg")
plt.show()
