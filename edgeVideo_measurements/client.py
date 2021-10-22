import os
import redis
import cv2
import json
import numpy as np
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
# Some basic setup:
# Setup detectron2 logger
import detectron2
import time
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
a_0 = np.arange(1,12,1).tolist()
a_1 = np.arange(13,26,1).tolist()
a_2 = [27,28]
a_3 = np.arange(31,45,1).tolist()
a_4 = np.arange(46,66,1).tolist()
a_5 = [67,70]
a_6 = np.arange(72,83,1).tolist()
a_7 = np.arange(84,91,1).tolist()
index_catId = a_0 + a_1 + a_2 + a_3 + a_4 +a_5 + a_6 + a_7
# 预测时ID与类别ID对应 https://blog.csdn.net/Dreaming_of_you/article/details/102756445
class mec_clien:
    def __init__(self, redis_host, client_id):
        self.conn = redis.Redis(host=redis_host, port=6379)
        self.id = client_id
        print('client start')
    # 遍历文件夹下所有图片
    def readFiles(self, file_path):
        filePath = '/home/lt/home/yison/code/mec/datasets/val2017'
        for dirpath, dirnames, filenames in os.walk(filePath):
            for file_name in filenames:
                data = cv2.imread(dirpath + '/' + file_name)
                msg = {
                    'id': file_name,
                    'data': data.tobytes()
                }
                self.putImg2Stream('test', msg)
    # 将图片加入到redis消息流中
    # msg :{ id: 'img_id', data: 'img_hex_data'}
    def putImg2Stream(self, stream_name, msg):
        self.conn.xadd(stream_name,  msg, '*', 10000)
    
# c = mec_clien('127.0.0.1', 'client_1')
# c.readFiles('/home/lt/home/yison/code/mec/datasets/val2017')




filePath = '/home/lt/home/yison/code/mec/datasets/val2017'
data = cv2.imread(filePath + '/' + '000000391144.jpg')
conn = redis.Redis(host='127.0.0.1', port=6379)
node_name = 'testTer'
conn.xadd("edge_node_1",  {'img': data.tobytes(), 'ter_name': node_name, 'image_id' : '391144', 'accuracy' : 2}, '*', 10000)
conn.delete(node_name)
conn.xadd(node_name,  {'image_id': '0', 'ter_name': node_name}, '*', 10000)
conn.xgroup_destroy(node_name, 'process_group')
conn.xgroup_create(node_name, 'process_group', id=0)
pic_res = []
while True:
    items = conn.xreadgroup('process_group', node_name+'_consumer', {node_name: '>'}, block=0, count=1)  

    res = items[0][1][0][1]
    msg_id = items[0][1][0][0].decode()
    print(msg_id)
    # conn.xack(node_name, 'processs_group', msg_id)
    if items[0][1][0][1][b'image_id'] == b'0':
        continue
    if items[0][1][0][1][b'image_id'] == b'1':
        break
    res_obj = {'image_id':int.from_bytes(res[b'image_id'],byteorder='little',signed=True),'category_id': int.from_bytes(res[b'category_id'],byteorder='little',signed=True),'bbox': np.frombuffer(res[b'bbox'], dtype=np.float32).tolist(),'score': float(res[b'score'].decode())}
    print(res_obj)
    pic_res.append(res_obj)
model = json.dumps(pic_res)
print('done')
os.remove('res.json')
fr = open('res.json', 'a')
fr.write(model)  
fr.close()

# # # success,encoded_image = cv2.imencode(".jpg", data.tobytes())
# # outputs = predictor(encoded_image)
# b_arr = data.tobytes()
# n_a = np.frombuffer(b_arr, dtype="uint8")
# n_a.shape = (427, 640, 3)
# success,encoded_image = cv2.imencode(".jpg", n_a)
# register_coco_instances("my_dataset_train", {}, "/home/lt/home/yison/datasets/coco/annotations/instances_train2017.json", "/home/lt/home/yison/datasets/coco/train2017")
# register_coco_instances("my_dataset_val", {}, "/home/lt/home/yison/datasets/coco/annotations/instances_val2017.json", "/home/lt/home/yison/datasets/coco/val2017")
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("my_dataset_train")
# cfg.MODEL.DEVICE = "cpu"
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
# predictor = DefaultPredictor(cfg)
# filePath = '/home/lt/home/yison/code/mec/datasets/val2017'
# fr = open('res.json', 'a')
# fr.write('[')
# for dirpath, dirnames, filenames in os.walk(filePath):
#     for file_name in filenames:
#         print(int(file_name[0:-4]))
#         data = cv2.imread(dirpath + '/' + file_name)
#         outputs = predictor(data)
#         print(outputs["instances"].pred_classes)
#         print(outputs["instances"].pred_boxes)
#         print(outputs["instances"].scores)
#         box_num = len(outputs["instances"].scores)
#         print(box_num)

#         for i in range(0, box_num):
#             r = {'image_id':int(file_name[0:-4]),'category_id': index_catId[outputs["instances"].pred_classes[i].item()],'bbox': outputs["instances"].pred_boxes[i].tensor[0].cpu().numpy().tolist(),'score': outputs["instances"].scores[i].item()}

#             print(r)
#             model=json.dumps(r)
#             fr.write(model)  
#             fr.write(','+'\n')
# fr.write(']')
# fr.close()
# print(outputs['instances'].pred_boxes)
# coco = COCO("/home/lt/home/yison/datasets/coco/annotations/instances_val2017.json")
# imgIds = coco.getImgIds() # 获取所有的image id，可以选择参数 coco.getImgIds(imgIds=[], catIds=[])
# imgIds = coco.getImgIds(imgIds=[0, 1, 2]) # 获得image id 为 0，1，2的图像的id
# imgIds = coco.getImgIds(catIds=[0, 1, 2]) # 获得包含类别 id 为0，1，2的图像
 
# annIds = coco.getAnnIds(catIds=[0, 1, 2]) # 获得类别id为0，1，2的标签
# annIds = coco.getAnnIds() # 获得和image id对应的标签
 
# catIds = coco.getCatIds(catNms=['0']) # 通过类别名筛选
# catIds = coco.getCatIds(catIds=[0, 1, 2]) # 通过id筛选
# catIds = coco.getCatIds(supNms=[]) # 通过父类的名筛选
# print('类别信息')
# cats_name = coco.loadCats(ids=catIds)
# print(cats_name)
 
# print('\n标签信息:')
# anns = coco.loadAnns(annIds)
# bboxes = np.array([i['bbox'] for i in anns]).astype(np.int32)
# cats = np.array([i['category_id'] for i in anns])
# print(anns)
# print('\n从标签中提取的Bounding box:')
# print(bboxes)
 
# print('图像')
# imgIdx = imgIds[0]
# img = coco.loadImgs([imgIdx]) # 读取图片信息
# print(annIds)
# annType = ['segm','bbox','keypoints']
# annType = annType[1]      #specify type here
# # # prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
# # # print ('Running demo for *%s* results.'%(annType))
# cocoGt=COCO("/home/lt/home/yison/datasets/coco/annotations/instances_val2017.json")
# cocoDt=cocoGt.loadRes('/home/lt/home/yison/code/mec/res.json')

# # imgIds=sorted(cocoGt.getImgIds())
# # print(imgIds)
# cocoEval = COCOeval(cocoGt,cocoDt,annType)
# cocoEval.params.imgIds  = [391144]
# cocoEval.evaluate()
# cocoEval.accumulate()
# a = cocoEval.summarize()
# print('--------------')
# print(cocoEval.stats[0])