import gym
import numpy as np
import json
import time
import math
from datetime import datetime
from skimage.transform import resize
from detectron2.config import get_cfg
from scipy import misc
import os
from detectron2.engine import DefaultPredictor
import redis
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from detectron2 import model_zoo
import random
ACTION_NUMS = 7 # action
ACTION_SPACE = {'Offload-Decision': [i for i in range(0, ACTION_NUMS)] } 
redis_conn = redis.Redis(host='127.0.0.1', port=6379)
TESTING = 0
TRAINING = 1
PIC_NAMES = []
with open('pic_names.json','r') as f:
    PIC_NAMES = json.load(f)
max_model_pricision = 0.2
cocoGt=COCO("/home/lt/home/yison/datasets/coco/annotations/instances_val2017.json")
user_model_list = ['mask_rcnn_R_50_C4_3x.yaml', 'faster_rcnn_R_101_DC5_3x.yaml', 'faster_rcnn_R_101_FPN_3x.yaml']
a_0 = np.arange(1,12,1).tolist()
a_1 = np.arange(13,26,1).tolist()
a_2 = [27,28]
a_3 = np.arange(31,45,1).tolist()
a_4 = np.arange(46,66,1).tolist()
a_5 = [67,70]
a_6 = np.arange(72,83,1).tolist()
a_7 = np.arange(84,91,1).tolist()
index_catId = a_0 + a_1 + a_2 + a_3 + a_4 +a_5 + a_6 + a_7
# the picture queue
def getPicFiles():
    filePath = '/home/lt/home/yison/code/mec/datasets/val2017'
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


class AtariWrapper:
    """Wraps over an Atari environment from OpenAI Gym and preprocesses observations."""

    def __init__(self, env_name, mode, action_space=None):
        """Creates the wrapper.

        Args:
            env_name: Name of an OpenAI Gym Atari environment.
            action_space: A list of possible actions. If 'action_space' is 'None' and no default
                configuration exists for this environment, all actions will be allowed.
            mode: The context in which the environment is used. Can be either environment.TESTING or
                environment.TRAINING.
        """

        if mode is not TESTING and mode is not TRAINING:
            raise ValueError(('Mode is invalid. Must be either environment.TESTING or '
                              'environment.TRAINING.'))

        self.env = gym.make(env_name)
        self.mode = mode
        self.observation_space = [47, 47, 1]
        self.reset()

        if action_space:
            self.action_space = list(action_space)
        elif env_name in ACTION_SPACE:
            self.action_space = ACTION_SPACE[env_name]
        else:
            self.action_space = list(range(self.env.action_space.n))

    def reset(self):
        """Resets the environment."""

        self.done = False
        self.episode_reward = 0
        self.episode_length = 0
        self.state = _preprocess_observation(self.env.reset())
        self.episode_start_time = time.time()
        self.episode_run_time = 0
        self.lives = None

    def step(self, action):
        """Performs the specified action.

        Returns:
            A reward signal which is either -1, 0 or 1.

        Raises:
            Exception: If the game ended.
            ValueError: If the action is not valid.
        """

        if self.done:
            raise Exception('Game finished.')

        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action,
                                                                                 self.action_space))

        observation, reward, self.done, info = self.env.step(action)

        if self.mode is TRAINING and self.lives is not None and info['ale.lives'] < self.lives:
            # While training, treat loss of life as end of episode.
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1
        self.state = _preprocess_observation(observation)
        self.episode_run_time = time.time() - self.episode_start_time
        self.lives = info['ale.lives']

        return -1 if reward < 0 else 1 if reward > 0 else 0

    def render(self):
        """Draws the environment."""

        self.env.render()

    def sample_action(self):
        """Samples a random action."""

        return np.random.choice(self.action_space)

    def get_state(self):
        """Gets the current state.

        Returns:
            An observation (47x47x1 tensor with float32 values between 0 and 1).
        """

        return self.state
# environment
class Env:
    
    def __init__(self,
                 n_edge=6,  
                 n_user=1  
                 ):
        self.n_edge = n_edge
        self.observation_space = [1, 31]
        self.action_space = ACTION_SPACE['Offload-Decision']
        self.n_user = n_user
        self.env_step = 0
        
        # self.longitude1 = 116.0
        # self.longitude2 = 116.8
        # self.latitude1 = 39.5
        # self.latitude2 = 40.3

        # user
        self.users = []
        for i in range(self.n_user):
            Open_id = np.random.randint(0, 10000)
            # print("Open_id"+str(Open_id))
            self.users.append(User(user_id=Open_id, task_queue_cap=250))
            self.users[i].user_location_list_, self.users[i].user_speed_list_ = user_trajectory(Open_id)

        self.reset()

    def reset(self):
        self.state = np.zeros((1, 32))
        self.users[0].reset()
        self.done = False
        self.env_step = 0
    def step(self, action): 
        reward = 0
        self.env_step = self.env_step + 1
        user = self.users[0]
        required_accuracy = user.img_accuracy # accuracy
        pic_accuracy, time_used, cpu_time_used = user.user_step([action, 0]) # Picture accuracy, program running time, CPU running time (energy consumption)
        reward = pic_accuracy/required_accuracy*1000 - time_used - cpu_time_used                # Accuracy normalization-delay-energy consumption The higher the better
        if reward < 0:
            reward = 0
        if self.env_step == 250:
            self.done =  True
        return reward
    def get_state(self):
        state = []
        user = self.users[0]
        user_location = user.user_location_list_[self.env_step]
        state.append(user_location[0])
        state.append(user_location[1])
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
        return state
    # Replace the node cache model
    def update_mec_node(self, action):
        return
    def sort_index(self, list_, element):
        element_index = 0
        if element in list_:
            list_ = sorted(list_)
            element_index = list_.index(element)
        else:
            print("None")
        return element_index


# 用户本地就缓存一个模型
class User:
    def __init__(self,
                 user_id=0,  # 用户id（从1开始，没有0）
                 task_queue_cap=20  # 用户有多少图片需要处理
                  ):
        self.task_queue_cap = task_queue_cap
        self.user_id = user_id
        self.pic_queue = PIC_NAMES[0:self.task_queue_cap]
        self.current_step = 0
        self.res_stream_name = str(self.user_id) + '_res_stream'
        self.predictor_list = []
        self.user_location_list_ = []
        self.res_file_name = 'res' + str(self.user_id) + '.json'
        self.user_speed_list_ = []
        self.conn = redis.Redis(host='127.0.0.1', port=6379)
        self.local_accuracy = 5
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
    # 执行用户step
    def user_step(self, rcv_act):
        pic_name = self.pic_queue[self.current_step]
        print(pic_name)
        data = cv2.imread(pic_name)
        pic_shape = data.shape
        pic_id = int(pic_name[-16:-4])
        remote_name = ''
        action = rcv_act[0] # 节点
        model_id = rcv_act[1] # 模型
        if action==0: # 本地处理
        # 本地精度不达标
            if self.img_accuracy > self.local_accuracy:
                return 0, 0, 0
            return self.porcess_local(data, pic_id,model_id)
        else:
            if action == 7: # 云端处理
                remote_name = 'cloud'
            else:
                remote_name = 'edge_node_' + str(action)
            self.conn.xadd(remote_name,  {'img': data.tobytes(), 'ter_name': self.res_stream_name, 'image_id' : pic_id, 'model_id': model_id, 'accuracy': self.img_accuracy, 'width': pic_shape[1], 'height': pic_shape[0]}, '*', 10000)
            start_time = time.time()
            cpu_start_time = time.clock()
            pic_res = []
            print('send ' + remote_name)
            while True:
                items = self.conn.xreadgroup('process_group', self.res_stream_name, {self.res_stream_name: '>'}, block=50000, count=1)  
                if len(items) == 0: # 超时
                    return 0, 0, 0
                res = items[0][1][0][1]
                msg_id = items[0][1][0][0].decode()
                # conn.xack(node_name, 'processs_group', msg_id)
                if res[b'image_id'] == b'0':
                    continue
                if res[b'image_id'] == b'1':
                    break
                if res[b'image_id'] == b'-1':
                    print('-1')
                    return 0, 0, 0
                res_obj = {'image_id':int.from_bytes(res[b'image_id'],byteorder='little',signed=True),'category_id': int.from_bytes(res[b'category_id'],byteorder='little',signed=True),'bbox': np.frombuffer(res[b'bbox'], dtype=np.float32).tolist(),'score': float(res[b'score'].decode())}
                print(res_obj)
                pic_res.append(res_obj)
            stop_time = time.time()
            cpu_stop_time = time.clock()
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
            cpu_time_used = cpu_stop_time - cpu_start_time # CPU运行差值 表示功耗
            self.current_step = self.current_step + 1
            return pic_precision, time_used, cpu_time_used
    def porcess_local(self, data, pic_id, model_id):
        cpu_start_time = time.clock()
        start_time = time.time()
        predictor_entity = self.predictor_list[model_id]
        outputs = predictor_entity(data)
        stop_time = time.time()
        cpu_stop_time = time.clock()
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
        return pic_precision, time_used, cpu_time_used
    def getOffloadRes(self, pic_id):
        cocoDt=cocoGt.loadRes(self.res_file_name)
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.params.imgIds  = [pic_id]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0]
    # user_location的获取和设置方法
    def get_user_location(self):
        return self.user_location

    def set_user_location(self, location):
        self.user_location = location

    # user的speed获取和设置方法
    def get_user_speed(self):
        return self.speed

    def set_user_speed(self, speed):
        self.speed = speed
    # 随机产生当前图片所需的精度
    def get_current_pricision(self):
        self.img_accuracy = random.uniform(0, max_model_pricision)
        return self.img_accuracy
    def reset(self):
        # 创建消息接收stream
        redis_conn.delete(self.res_stream_name)
        redis_conn.xadd(self.res_stream_name,  {'image_id': '0', 'ter_name': self.res_stream_name} ,'*', 10000)
        redis_conn.xgroup_destroy(self.res_stream_name, 'process_group')
        redis_conn.xgroup_create(self.res_stream_name, 'process_group', id=0)
        # 将当前待处理图片队列指针归零
        self.current_step = 0
        
        

class EdgePoint:  # 边缘点
    def __init__(self,
                 ep_location=None,  # EdgePoint的位置
                 ep_capability_ran=10e6,  # 总RAN切片的网络能力Gb/s?
                 ep_capability_mec=10e9,  # 总MEC服务器的计算能力GHz
                 num_slice=21,  # 网络能力和MEC服务器计算资源的切片数量k
                 ):
        self.ep_location = ep_location
        self.ep_capability_ran = ep_capability_ran
        self.ep_capability_mec = ep_capability_mec
        self.num_slice = num_slice
        self.ran_slicing_dec = [1] * self.num_slice # EdgePoint上RAN网络资源切片决策(初始化为1)
        self.mec_slicing_dec = [1] * self.num_slice  # EdgePoint上MEC计算资源切片决策
        self.users_list = []  # 用于存放卸载到该边缘点的用户
        self.ran_slicing_list = []  # 切片好的RAN资源
        self.mec_slicing_list = []  # 切片好的ME资源

    def reset(self):
        self.users_list = []

class Cloud:
    def __init__(self,
                 cloud_capability_comp=150e9,  # 远程云的计算能力
                 cloud_capability_net=1e6,  # 远程云的网络能力
                 cloud_dist=1e5,  # 云到用户的距离
                 cloud_R = 1e9 #用户到远程云的信道传输能力，比较小
                 ):
        self.cloud_capability_comp = cloud_capability_comp
        self.cloud_capability_net = cloud_capability_net
        self.cloud_dist = cloud_dist
        self.cloud_R = cloud_R

