import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pprint

IMAGES_PATH = '/root/Dataset/fcn90_pred/psp_resnet50_sem'  # 图片集地址
IMAGES_FORMAT = ['.png', '.PNG']  # 图片格式
IMAGE_SIZE = (640, 480) # 每张小图片的大小
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = '/root/Dataset/fcn90_pred/pred_recombination'  # 图片转换后的地址


''' 建个字典pictures 用于存放所有的图片和其字图片， {图片名：字图片绝对地址列表} '''
image_path = [os.path.join(IMAGES_PATH, name) for name in sorted(os.listdir(IMAGES_PATH))]    # 图片路径
image_names = [name for name in sorted(os.listdir(IMAGES_PATH))]                           # 图片名称
# print(image_names)
names = list(Counter([img.split('.')[0].split('_')[0] +'_'+  img.split('.')[0].split('_')[1]  for img in image_names]).keys())               # 图片名称统计
# print(names) +'_'+  image.split('.')[0].split('_')[2]
pictures = {}
for name in names:
    pictures[name] = []
    for img in image_path:
        if os.path.basename(img).startswith(name):
            pictures[name].append(img)
pprint.pprint(pictures)

''' 定义图像拼接函数 '''
def image_compose(pic_name, pic_path):
    # 简单的对于参数的设定和实际图片集的大小进行数量判断
    if len(pic_path) != IMAGE_ROW * IMAGE_COLUMN:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE[0], IMAGE_ROW * IMAGE_SIZE[1]))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
       for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(pic_path[IMAGE_COLUMN * (y - 1) + x - 1])
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE[0], (y - 1) * IMAGE_SIZE[1]))
    return to_image.save(os.path.join(IMAGE_SAVE_PATH, pic_name)+'.png')  # 保存新图

for pic_name, pic_path in pictures.items():
    image_compose(pic_name, pic_path)

