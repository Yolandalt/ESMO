from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import redis
import cv2
import numpy as np
import argparse
import json
PARSER = argparse.ArgumentParser(description='Start a new node.')

PARSER.add_argument('--node_name',
                    metavar='edgeNode1',
                    help='name of edge node',
                    default='edgeNode0')
PARSER.add_argument('--redis_host',
                    metavar='127.0.0.1',
                    help='host of redis',
                    default='127.0.0.1')
class processNode:
    def __init__(self, redis_host, node_name, model_name):
        self.conn = redis.Redis(host=redis_host, port=6379)
        self.predictors = []
        file_names = []
        with open('config_name.json', 'r') as f:
            file_names = json.load(f)
        for config_name in file_names:
            print(config_name)
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/' + config_name))
            cfg.MODEL.DEVICE = "cpu"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/' + config_name)  # Let training initialize from model zoo
            predictor = DefaultPredictor(cfg)
            self.predictors.append(predictor)
        self.conn.delete(node_name)
        self.conn.xadd(node_name,  {'img': 'create', 'ter_name': 'testTer', 'image_id': '0'}, '*', 10000)
        self.conn.xgroup_destroy(node_name, 'process_group')
        self.conn.xgroup_create(node_name, 'process_group', id=0)  # 0 从开始消费, $ 从尾部消费
        self.name = node_name
        a_0 = np.arange(1,12,1).tolist()
        a_1 = np.arange(13,26,1).tolist()
        a_2 = [27,28]
        a_3 = np.arange(31,45,1).tolist()
        a_4 = np.arange(46,66,1).tolist()
        a_5 = [67,70]
        a_6 = np.arange(72,83,1).tolist()
        a_7 = np.arange(84,91,1).tolist()
        self.index_catId = a_0 + a_1 + a_2 + a_3 + a_4 +a_5 + a_6 + a_7
    def getImgsAndTrain(self):
        print(self.name + ' start working')
        ter_name = ''
        while True:
            items = self.conn.xreadgroup('process_group', self.name+'_consumer', {self.name: '>'}, block=0, count=1)  
            pic_data = items[0][1][0][1][b'img']
            # success,encoded_image = cv2.imdecode(".jpg", pic_data)
            pic_data_np = np.frombuffer(pic_data, 'uint8')
            if pic_data_np.shape[0] == 6:
                continue
            pic_width =  int(items[0][1][0][1][b'width'])
            pic_height = int(items[0][1][0][1][b'height'])
            print(pic_width)
            print(pic_data_np.shape)
            id = items[0][1][0][1][b'image_id']
            ter_name = items[0][1][0][1][b'ter_name']
            accuracy = float(items[0][1][0][1][b'accuracy'].decode())
            required_models = json.loads(self.conn.hget('edge_models', self.name))
            model_accuracys = {}
            for i in required_models:
                model_accuracys[i]=self.conn.hget('model_accuracy', str(i)).decode()
            d_order=sorted(model_accuracys.items(),key=lambda x:x[1],reverse=False) # 按照精度由小到大排序
            # 选择最小满足精度的
            predictor_index = -1
            for index in d_order:
                if float(index[1]) >= accuracy:
                    predictor_index = index[0]
            if predictor_index == -1: # 此处精度不达标
                accuracy_negetive_msg = {'image_id':-1}
                self.conn.xadd(ter_name,  accuracy_negetive_msg, '*', 10000)
                continue
            print(pic_data_np)
            pic_data_np = pic_data_np.reshape(pic_height, pic_width, 3)
            print(predictor_index)
            outputs = self.predictors[predictor_index](pic_data_np)
            box_num = len(outputs["instances"].scores)
            print(outputs)
            for i in range(0, box_num):
                res = {'image_id':int(id).to_bytes(4, byteorder='little', signed=True),'category_id': self.index_catId[outputs["instances"].pred_classes[i].item()].to_bytes(4, byteorder='little', signed=True),'bbox': outputs["instances"].pred_boxes[i].tensor[0].cpu().numpy().tobytes(),'score': outputs["instances"].scores[i].item()}
                print(res)
                self.conn.xadd(ter_name,  res, '*', 10000)
            process_over_msg = {'image_id':1,'category_id': self.index_catId[outputs["instances"].pred_classes[i].item()],'bbox': outputs["instances"].pred_boxes[i].tensor[0].cpu().numpy().tobytes(),'score': outputs["instances"].scores[i].item()}
            self.conn.xadd(ter_name, process_over_msg, '*', 10000)

def main(args):
    node = processNode(args.redis_host, args.node_name, 'mask_rcnn_R_50_FPN_1x.yaml')
    node.getImgsAndTrain()

if __name__ == '__main__':
    main(PARSER.parse_args())