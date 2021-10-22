# 整合数据
import json

def readData(dir):
    with open(dir, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict
print(load_dict[0].get("pic_id"))


