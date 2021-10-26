import numpy as np
from EdgeVideoClass.Caching import ModelCaching
from EdgeVideoClass.User import User_class

import json

class TR_Class:
    def __init__(self,
        n_edge=6,
        n_user = 5,
        n_CNNmodel = 8,
                 ):
        self.__n_edge = n_edge
        self.__n_user = n_user
        self.__n_CNNmodel = n_CNNmodel
        # Define the object of the class
        self.cache = ModelCaching(n_edge=self.__n_edge, n_CNNModel=self.__n_CNNmodel)
        # Create user
        self.Users = []
        for i in range(self.__n_user):
            self.Users.append(User_class(task_arrival_rate=0.5, energy_unit=5e-5, n_edge=self.__n_edge))

        # Initialize the model cache decision variable n_edge*n_CNNModel matrix (value is 0 or 1)
        self.caching_decision_all = self.cache.init_caching_decision()

        # Weight
        self.belta = 0.5
        # There are different CNN models
        # cloud
        self.TR_Result_cloud = []
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_C4_1x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_C4_3x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_DC5_3x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_50_FPN_1x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_C4_3x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_DC5_3x.json"))
        self.TR_Result_cloud.append(self.ReadTRResults("./TargetRecognition/data/102/mask_rcnn_R_101_FPN_3x.json"))
        # local
        self.TR_Result_local = self.ReadTRResults("./TargetRecognition/data/103/mask_rcnn_R_50_C4_1x.json")-

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

    # Get the total length of the chromosome (i.e., all users and all frames)
    def getEncodedLength(self):
        lengthEncode = 0
        for user in range(self.__n_user):
            lengthEncode += self.Users[user].n_frames
        return lengthEncode

    # Determine the chromosome length of the cache decision variable Y
    def getEncodedLength_Y(self):
        # There is a cache decision variable on each edge node, and for each edge node, the serial number of the model is stored
        lengthEncode_Y = 0
        # Sum of decision variables
        for edge in range(self.__n_edge):
            lengthEncode_Y += self.cache.num_CNN_in_edge[edge]
        return lengthEncode_Y

    # Chromosome decoding to get phenotypic solution
    # The phenotype here is the unloading decision variable, that is, the frame corresponding to the n_user* user,
    # and the corresponding value is the label of the computing node
    def decodedChromosome(self, chromosome):
        # Initial unloading decision n_user*n_frame, for each frame of each user, 0 is local;
        # 1-6 are edge nodes; 7 is remote cloud
        offloading_decision = []
        # Start decoding
        # Take a chromosome
        start_index = 0
        for user in range(self.__n_user):
            end_index = start_index + self.Users[user].n_frames
            # Take a chromosome fragment of a user
            chromosome_slice = chromosome[start_index:end_index]
            offloading_decision.append(chromosome_slice)
            start_index += self.Users[user].n_frames
        return offloading_decision

    # Chromosome decoding to get phenotypic solution
    # The phenotype here is the number of CNN models corresponding
    # to the cache decision variable, populationSize*n_edge
    def decodedChromosome_Y(self, chromosome_Y):
        # Initialize the cache decision n_edge*n_CNNmodel,
        # for the CNN model on each edge node, 0 means cache, 1 means no cache
        decoded_caching_decision = []
        # Start decoding
        # For every edge node
        for edge in range(self.__n_edge):
            decoded_caching_decision.append(chromosome_Y[edge*self.__n_CNNmodel:(edge+1)*self.__n_CNNmodel])
        return decoded_caching_decision

    def decodedChromosome_both(self,chromosome):
        # chromosome: The first is the model cache decision,
        # and the latter is the calculation offload decision
        lengthEncode_Caching = self.getEncodedLength_Y()
        lengthEncode_offloading = self.getEncodedLength()
        chromosome_Y = chromosome[:lengthEncode_Caching]
        chromosome_X = chromosome[lengthEncode_Caching:]
        # Model cache
        decoded_caching_decision = []*self.__n_edge
        for edge in range(self.__n_edge):
            decoded_caching_decision.append(np.zeros(self.__n_CNNmodel))
        # Start decoding
        # For every edge node
        last_length_model = 0
        last_edge_num_CNN = 0
        for edge in range(self.__n_edge):
            last_edge_num_CNN = self.cache.num_CNN_in_edge[edge]
            for model in range(last_edge_num_CNN):
                index_model = last_length_model+int(chromosome_Y[model])
                decoded_caching_decision[edge][index_model] = 1
        last_length_model += last_edge_num_CNN
        # computation offload
        decoded_offloading_decision = []
        # Start decoding
        # Take a chromosome
        start_index = 0
        for user in range(self.__n_user):
            end_index = start_index + self.Users[user].n_frames
            # Take a chromosome fragment of a user
            chromosome_slice = chromosome[start_index:end_index]
            decoded_offloading_decision.append(chromosome_slice)
            start_index += self.Users[user].n_frames
        return decoded_caching_decision, decoded_offloading_decision

    # Find out how many users (number of users) are cached on an edge node.
    # The total number of frames required by the main requirement
    # In fact, it should be asked to uninstall the frame on the edge node! Remember to optimize
    def get_users_in_edge(self, offloading_decision):
        # initialize n_edge
        users_in_edges = [ [] for j in range(self.__n_edge) ]
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                for edge in range(self.__n_edge):
                    if offloading_decision[user][frame] == edge+1:
                        users_in_edges[edge].append(self.Users[user])
        return users_in_edges

    # Calculate the fitness of local computing
    def get_local_fitness_value(self, frame_id):
        # fitness_values_for_local = 0
        # accuracy
        pic_precision = self.TR_Result_local[frame_id].get("pic_precision")
        # delay
        delay = self.TR_Result_local[frame_id].get("time_used")
        # energy
        energy_cost = self.TR_Result_local[frame_id].get("cpu_time_used")
        energy_addition = np.random.uniform(20, 25)
        # compute fitness value
        fitness_values_for_local = delay + self.belta*energy_cost+energy_addition

        return fitness_values_for_local,pic_precision

    # Calculate the fitness of remote cloud
    def get_cloud_fitness_value(self, frame_id):
        pic_precision_all_model = []
        for model in range(len(self.TR_Result_cloud)):
            # accuracy
            pic_precision_all_model.append(self.TR_Result_cloud[model][frame_id].get("pic_precision"))
        # max accuracy
        pic_precision_max = max(pic_precision_all_model)
        select_model = self.get_index1(pic_precision_all_model,pic_precision_max)[0]
        # delay
        delay = self.TR_Result_cloud[select_model][frame_id].get("time_used")
        # energy
        energy_cost = self.TR_Result_cloud[select_model][frame_id].get("cpu_time_used")
        # Transmission delay
        delay_addition = np.random.uniform(20, 40)
        # compute fitness value
        fitness_values_for_cloud = delay + self.belta * energy_cost+delay_addition
        return fitness_values_for_cloud, pic_precision_max

    # Calculate the fitness of edge nodes
    def get_edge_fitness_value(self, frame_id, edge_id,model_id):
        # Obtain the optimal computing power of the edge
        # accuracy
        pic_precision = self.TR_Result_edge_all[edge_id][model_id][frame_id].get("pic_precision")
        # delay
        delay = self.TR_Result_edge_all[edge_id][model_id][frame_id].get("time_used")
        # energy
        energy_cost = self.TR_Result_edge_all[edge_id][model_id][frame_id].get("cpu_time_used")
        # compute fitness value
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

        return fitness_values_for_edge,pic_precision

    def get_one_chromosome_fitness(self, decoded_offloding_decison,decoded_caching_decision):
        # users_in_edges_one: Users of all nodes on a chromosome
        fitness_value_one = 0
        offloading_decision = decoded_offloding_decison
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                if offloading_decision[user][frame] == 0:  # local
                    # Integration of local processing delay and local energy consumption
                    one, _ = self.get_local_fitness_value(frame_id)
                    fitness_value_one += one
                elif offloading_decision[user][frame] == self.__n_edge + 1:  # remote cloud
                    one, _ = self.get_cloud_fitness_value(frame_id)
                    fitness_value_one += one
                else:  # offloading to the edge node
                    # Need to judge whether there is a CNN model that meets the accuracy
                    # First find out the CNN model cached on the node (find out all subscripts with element 1)
                    edge = int(offloading_decision[user][frame])-1
                    a = decoded_caching_decision[edge]
                    CNN_list = [i for i in range(len(a)) if a[i] == 1]

                    if CNN_list:
                        # Take out the corresponding accuracy
                        accuray_caching = []
                        for i in range(len(CNN_list)):
                            accuray_model = self.TR_Result_edge_all[edge][CNN_list[i]][frame_id].get("pic_precision")
                            accuray_caching.append(accuray_model)
                        # Take the maximum value
                        max_accuracy_CNN = max(accuray_caching)
                        corresponding_model_index = self.get_index1(accuray_caching,max_accuracy_CNN)[0]
                        model_index = CNN_list[corresponding_model_index]
                        one, _ = self.get_edge_fitness_value(frame_id, edge, model_index)
                        fitness_value_one += one
                    else:
                        # offloading to cloud
                        one, _ = self.get_cloud_fitness_value(frame_id)
                        fitness_value_one += one
        return fitness_value_one

    def get_accuracy_fitnessValue(self, decoded_offloading_decision, decoded_model_caching_decision):
        accuracy_fitnessValue = 0
        for user in range(len(decoded_offloading_decision)):
            for frame in range(len(decoded_offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                if decoded_offloading_decision[user][frame] == 0:   # local
                    # Take the lowest accuracy of the model
                    _, one = self.get_local_fitness_value(frame_id)
                    accuracy_fitnessValue += one
                elif decoded_offloading_decision[user][frame] == self.__n_edge + 1:  # remote cloud
                    _, one = self.get_cloud_fitness_value(frame_id)
                    accuracy_fitnessValue += one
                else:  # offloading to the edge node
                    # Need to judge whether there is a CNN model that meets the accuracy
                    # First find out the CNN model cached on the node (find out all subscripts with element 1)
                    edge = int(decoded_offloading_decision[user][frame])-1
                    a = decoded_model_caching_decision[edge]
                    CNN_list = [i for i in range(len(a)) if a[i] == 1]

                    if CNN_list:
                        accuray_caching = []
                        for i in range(len(CNN_list)):
                            accuray_model = self.TR_Result_edge_all[edge][CNN_list[i]][frame_id].get("pic_precision") * 10
                            accuray_caching.append(accuray_model)
                        # Take the maximum value
                        max_accuracy_CNN = max(accuray_caching)
                        corresponding_model_index = self.get_index1(accuray_caching,max_accuracy_CNN)[0]
                        _, one = self.get_edge_fitness_value(frame_id,edge,corresponding_model_index)
                        accuracy_fitnessValue += one
                    else:
                        # offloading to cloud
                        _, one = self.get_cloud_fitness_value(frame_id)
                        accuracy_fitnessValue -= one
                        # accuracy_fitnessValue += np.max(self.cache.accuracy_provide)
        return accuracy_fitnessValue

    def get_one_chromosome_fitness_local(self, decoded_offloding_decison,decoded_caching_decision):
        # users_in_edges_one Users of all nodes on a chromosome
        fitness_value_one = 0
        offloading_decision = decoded_offloding_decison
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                # only local
                # Integration of local processing delay and local energy consumption
                one, _ = self.get_local_fitness_value(frame_id)
                fitness_value_one += one
        return fitness_value_one

    def get_accuracy_fitnessValue_local(self, decoded_offloading_decision, decoded_model_caching_decision):
        accuracy_fitnessValue = 0
        for user in range(len(decoded_offloading_decision)):
            for frame in range(len(decoded_offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                # lcoal
                # Take the lowest accuracy of the model
                _, one = self.get_local_fitness_value(frame_id)
                accuracy_fitnessValue += one
        return accuracy_fitnessValue

    def get_one_chromosome_fitness_cloud(self, decoded_offloding_decison, decoded_caching_decision):
        # users_in_edges_one Users of all nodes on a chromosome
        fitness_value_one = 0
        offloading_decision = decoded_offloding_decison
        for user in range(len(offloading_decision)):
            for frame in range(len(offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                # lcoal
                # Take the lowest accuracy of the model
                one, _ = self.get_cloud_fitness_value(frame_id)
                fitness_value_one += one
        return fitness_value_one

    def get_accuracy_fitnessValue_cloud(self, decoded_offloading_decision, decoded_model_caching_decision):
        accuracy_fitnessValue = 0
        for user in range(len(decoded_offloading_decision)):
            for frame in range(len(decoded_offloading_decision[user])):
                frame_id = self.Users[user].frame_id[frame]
                # lcoal
                # Take the lowest accuracy of the model
                _, one = self.get_cloud_fitness_value(frame_id)
                accuracy_fitnessValue += one
        return accuracy_fitnessValue

    # Return all subscripts of an element in the list
    def get_index1(self, lst=None, item=0):
        return [index for (index, value) in enumerate(lst) if value == item]