# 7 CNN models, determine whether each CNN model exists on 5 edge nodes (y = 0 or 1)
import numpy as np

class ModelCaching:
    def __init__(self,
                 n_edge=6,
                 n_CNNModel=8,
                 ):
        self.n_edge = n_edge
        self.n_CNNModel = n_CNNModel
        # Define model caching decisions
        self.caching_decision_all = np.zeros((self.n_edge, self.n_CNNModel))
        # Initialize the buffer size required for each CNN model
        self.cache_require = np.random.uniform(1, 5, self.n_CNNModel)

        # Initialize the maximum buffer space of each MEC node
        self.cache_max_mec = np.random.uniform(5, 10, self.n_edge)
        # Cache space used on MEC node
        self.cache_mec = np.zeros(self.n_edge)
        # Define the number of CNN models cached on the model
        self.num_CNN_in_edge = np.random.randint(0, n_CNNModel, self.n_edge)

    # Determine whether the cache resources on the edge node of the given point are free
    def is_cache_full(self, edgeNode):
        # Consider the model cache on each edge node
        for j in range(self.n_CNNModel):
            if self.caching_decision_all[edgeNode][j] == 1:
                self.cache_mec[edgeNode] += self.cache_require[j]  # Add if already cached
        # After traversing all models, you can get the currently used cache resources
        if self.cache_mec[edgeNode] <= self.cache_max_mec[edgeNode]:
            # If it is less than, there is still room, return true
            return 1
        else:
            return 0

    # Initialize the model cache decision and need to meet resource constraints
    def init_caching_decision(self):
        # Operate the cache decision on each MEC node
        for i in range(self.n_edge):
            # Determine whether the cache resources on the node are sufficient
            while self.is_cache_full(i):
                # If enough, a CNN model is randomly selected for caching
                self.caching_decision_all[i][np.random.randint(0, self.n_CNNModel)] = 1
        return self.caching_decision_all











