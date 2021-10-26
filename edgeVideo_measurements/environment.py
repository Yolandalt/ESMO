import numpy as np
import json
import time
import math
from datetime import datetime
from detectron2.config import get_cfg
import os
from detectron2.engine import DefaultPredictor
import redis
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from detectron2 import model_zoo
import random
ACTION_NUMS = 8 # action
# action space
ACTION_SPACE = {'Offload-Decision': [i for i in range(0, ACTION_NUMS)] }
redis_conn = redis.Redis(host='127.0.0.1', port=6379)
TESTING = 0
TRAINING = 1
PIC_NAMES = []
with open('pic_names.json','r') as f:
    PIC_NAMES = json.load(f)
max_model_pricision = 0.2
cocoGt=COCO("/home/mec/mec/datasets/instances_val2017.json")
user_model_list = ['mask_rcnn_R_50_DC5_3x.yaml']
a_0 = np.arange(1,12,1).tolist()
a_1 = np.arange(13,26,1).tolist()
a_2 = [27,28]
a_3 = np.arange(31,45,1).tolist()
a_4 = np.arange(46,66,1).tolist()
a_5 = [67,70]
a_6 = np.arange(72,83,1).tolist()
a_7 = np.arange(84,91,1).tolist()
index_catId = a_0 + a_1 + a_2 + a_3 + a_4 +a_5 + a_6 + a_7
final_res = []
# Picture queue
def getPicFiles():
    filePath = '/home/mec/mec/datasets/val2017'
    for dirpath, dirnames, filenames in os.walk(filePath):
        for file_name in filenames:
            single_pic_name = filePath + '/' + file_name
            PIC_NAMES.append(single_pic_name)
def _preprocess_observation(observation):
    """Transforms the specified observation into a 47x47x1 grayscale image.

    Returns:
        A 47x47x1 tensor with float32 values between 0 and 1.
    """

    # Transform the observation into a grayscale image with values between 0 and 1. Use the simple
    # np.mean method instead of sophisticated luminance extraction techniques since they do not seem
    # to improve training.
    grayscale_observation = observation.mean(2)

    # Resize grayscale frame to a 47x47 matrix of 32-bit floats.
    # resized_observation = misc.imresize(grayscale_observation, (47, 47)).astype(np.float32)
    resized_observation = resize(grayscale_observation,output_shape=(47, 47))
    return np.expand_dims(resized_observation, 2)

# environment
class Env:
    
    def __init__(self,
                 n_edge=6,  # MEC
                 n_user=1,
                 index=0
                 ):
        self.n_edge = n_edge
        self.action_space = ACTION_SPACE['Offload-Decision']
        self.n_user = n_user
        self.env_step = 0
        self.episode_length = 0
        self.batch_size = 10
        self.index = index
        # redis_conn.lpush('reward' + str(index), '==')
        # redis_conn.lpush('time_used' + str(index), '==')
        # redis_conn.lpush('required_accuracy' + str(index), '==')
        # redis_conn.lpush('pic_accuracy' + str(index), '==')
        # redis_conn.lpush('cpu_used' + str(index), '==')
        self.observation_space = [self.batch_size, 38]

        # user
        self.users = []
        for i in range(self.n_user):
            Open_id = np.random.randint(0, 10000)
            self.users.append(User(user_id=Open_id, task_queue_cap=250, user_index=index))
        self.reset()

    def reset(self):
        print('-------------------------env reset--------------------------')
        self.state = np.zeros((1, 32))
        self.users[0].reset()
        self.done = False
        self.env_step = 0
        self.episode_reward = 0
    def step(self): 
        # Update model parameters at regular intervals
        reward = 0
        self.env_step = self.env_step + 1
        user = self.users[0]
        required_accuracy = user.img_accuracy # Get the required accuracy
        pic_accuracy, time_used, cpu_time_used, remote_time_used = user.user_step([0, 0]) # Picture accuracy, program running time, CPU running time (energy consumption)
        reward = time_used + cpu_time_used                # Accuracy normalization-delay-energy consumption The higher the better
        if reward < 0:
            reward = 0
        self.episode_reward += reward
        self.episode_length += 1
        if self.env_step == self.batch_size:
            self.done =  True
        print('----------------------------current reward------------------------------------------')
        print(reward)    

        return reward
    def get_state(self):
        state = []
        user = self.users[0]
        posi_len = len(user.user_location_list_)
        user_location = user.user_location_list_[self.env_step % posi_len]
        user.ul = user_location
        # location
        state.append(float(user_location[0]))
        state.append(float(user_location[1]))
        # accuracy
        state.append(user.get_current_pricision())
        # The node has not processed the image queue
        for i in range(1, 7):
            stream_name = 'edge_node_' + str(i)
            state.append(int(redis_conn.xlen(stream_name)))
        state.append(int(redis_conn.xlen('cloud')))
        for i in range(1, 7):
            stream_name = 'edge_node_' + str(i)
            model_list = json.loads(redis_conn.hget('edge_models', stream_name))
            for j in model_list:
                state.append(j)
        model_list = json.loads(redis_conn.hget('edge_models', 'cloud'))
        for j in model_list:
            state.append(j)
        for j in mp.mec_positions:
            state.append(get_distance(user_location[0], user_location[1], j[0], j[1]))
        print('-----------------------------------------current_state--------------------------------------------')
        print(state)
        return state
    # Replace the node cache model
    def update_mec_node(self, action):
        return
    # The list elements are sorted, and then the index of the corresponding element is retrieved
    def sort_index(self, list_, element):
        element_index = 0
        if element in list_:
            list_ = sorted(list_)
            element_index = list_.index(element)
        else:
            print("None")
        return element_index

# User
class User:
    def __init__(self,
                 user_id=0,  # user id
                 task_queue_cap=20,
                 user_index=0  # How many pictures the user needs to process
                  ):
        self.user_index = user_index
        self.task_queue_cap = task_queue_cap
        self.user_id = user_id
        self.pic_queue = PIC_NAMES[0:self.task_queue_cap]
        self.current_step = 0
        self.res_stream_name = str(self.user_index) + '_res_stream'
        self.predictor_list = []
        self.user_location_list_ = []
        self.res_file_name = 'res/res' + str(self.user_index) + '.json'
        self.user_speed_list_ = []
        self.conn = redis.Redis(host='127.0.0.1', port=6379)
        self.local_accuracy = 5
        self.ul = []
        self.img_accuracy = 0
        for i in range(0, 1):
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/' + user_model_list[i]))
            cfg.MODEL.DEVICE = "cpu"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/' + user_model_list[i])
            predictor = DefaultPredictor(cfg)
            self.predictor_list.append(predictor)
        self.reset()
    # user step
    def user_step(self, rcv_act):
        pic_name = self.pic_queue[self.current_step % 250]
        print(pic_name)
        data = cv2.imread(pic_name)
        pic_id = int(pic_name[-16:-4])
        remote_name = ''
        action = rcv_act[0] # node
        model_id = rcv_act[1] # model
        if action==0: # local processing
        # Local accuracy is not up to standard
            if self.img_accuracy > self.local_accuracy:
                return 100, 100, 100,0
            return self.porcess_local(data, pic_id,model_id)
        else:
            if action == 7: # cloud processing
                remote_name = 'cloud'
            else:
                remote_name = 'edge_node_' + str(action)
            self.conn.xadd(remote_name,  {'img': data.tobytes(), 'ter_name': self.res_stream_name, 'image_id' : pic_id, 'model_id': model_id, 'accuracy': self.img_accuracy, 'width': pic_shape[1], 'height': pic_shape[0]}, '*', 10000)
            start_time = time.time()
            print(start_time)
            cpu_start_time = time.perf_counter()
            print(cpu_start_time)
            pic_res = []
            print('send ' + remote_name)
            while True:
                items = self.conn.xreadgroup('process_group', self.res_stream_name, {self.res_stream_name: '>'}, block=0, count=1)  
                if len(items) == 0: # time out
                    return 100,100,100,100
                res = items[0][1][0][1]
                msg_id = items[0][1][0][0].decode()
                remote_time_used = 0
                if res[b'image_id'] == b'0':
                    continue
                if res[b'image_id'] == b'1':
                    remote_time_used = float(res[b'time_used'])
                    print(remote_time_used)
                    break
                if res[b'image_id'] == b'-1':
                    print('-1')
                    return 100, 100, 100,0
                res_obj = {'image_id':int.from_bytes(res[b'image_id'],byteorder='little',signed=True),'category_id': int.from_bytes(res[b'category_id'],byteorder='little',signed=True),'bbox': np.frombuffer(res[b'bbox'], dtype=np.float32).tolist(),'score': float(res[b'score'].decode())}
                pic_res.append(res_obj)
            stop_time = time.time()
            print(stop_time)
            cpu_stop_time = time.perf_counter()
            print(cpu_stop_time)
            model = json.dumps(pic_res)
            print(str(self.user_id) + ' ' + pic_name + ' done')
            if os.path.exists(self.res_file_name):
                os.remove(self.res_file_name)
            fr = open(self.res_file_name, 'a')
            fr.write(model)  
            fr.close()
            self.current_step = self.current_step + 1
            pic_precision = self.getOffloadRes(pic_id)
            time_used = stop_time - start_time
            if action==7:# 云
                time_used = time_used + random.randint(10,19) #Extended in the cloud
            if not action==0:
                time_used = time_used + get_distance(self.ul[0], self.ul[1], mp.mec_positions[action-1][0], mp.mec_positions[action-1][1])
            cpu_time_used = cpu_stop_time - cpu_start_time # CPU running difference indicates power consumption
            self.current_step = self.current_step + 1
            return pic_precision, time_used, cpu_time_used,remote_time_used
    def porcess_local(self, data, pic_id, model_id):
        cpu_start_time = time.perf_counter()
        start_time = time.time()
        predictor_entity = self.predictor_list[0]
        outputs = predictor_entity(data)
        stop_time = time.time()
        cpu_stop_time = time.perf_counter()
        box_num = len(outputs["instances"].scores)
        pic_res = []
        for i in range(0, box_num):
            res = {'image_id':pic_id,'category_id': index_catId[outputs["instances"].pred_classes[i].item()],'bbox': outputs["instances"].pred_boxes[i].tensor[0].cpu().numpy().tolist(),'score': outputs["instances"].scores[i].item()}
            pic_res.append(res)
        time_used = stop_time - start_time
        cpu_time_used = cpu_stop_time - cpu_start_time
        model = json.dumps(pic_res)
        if os.path.exists(self.res_file_name):
            os.remove(self.res_file_name)
        
        fr = open(self.res_file_name, 'a')
        fr.write(model)  
        fr.close()
        pic_precision = self.getOffloadRes(pic_id)
        obj = {}
        obj['pic_id'] = pic_id
        obj['pic_precision'] = pic_precision
        obj['time_used'] = time_used
        obj['cpu_time_used'] = cpu_time_used
        final_res.append(obj)
        self.current_step += 1
        return pic_precision, time_used, cpu_time_used,0
    def getOffloadRes(self, pic_id):
        cocoDt=cocoGt.loadRes(self.res_file_name)
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.params.imgIds  = [pic_id]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0]

    def get_user_location(self):
        return self.user_location

    def set_user_location(self, location):
        self.user_location = location

    def get_user_speed(self):
        return self.speed

    def set_user_speed(self, speed):
        self.speed = speed
    # Randomly generate the accuracy required for the current picture
    def get_current_pricision(self):
        self.img_accuracy = random.uniform(0.018, 0.031)
        return self.img_accuracy
    def reset(self):
        self.current_step = 0

# Find the distance between two points (latitude and longitude) (unit: km)
def get_distance(lon_a, lat_a, lon_b, lat_b):
    radlat1 = math.radians(float(lat_a))
    radlat2 = math.radians(float(lat_b))
    a = radlat1 - radlat2
    b = math.radians(float(lon_a)) - math.radians(float(lon_b))
    s = 2 * math.asin(math.sqrt(pow(math.sin(a/2),2) + math.cos(radlat1) * math.cos(radlat2)*pow(math.sin(b/2),2)))
    earth_radius = 6378.137
    s = s * earth_radius
    return s  # m

# Running
e = Env()
for i in range(0, 100):
    print(i)
    e.step()
with open(user_model_list[0].split('.')[0] + '.json',"w") as f:
    json.dump(final_res,f)
