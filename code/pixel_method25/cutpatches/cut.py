'''
    将原图切分成patch并保存
'''
import  sys
sys.path.append("/root/SEM/pixel_method25")
from dataset.image_dataset import ImageDataset
from cutpatches.getpatches import extract_patches
from config import param
import os

''' *************************读取原图、 创建Dataset************************* '''

img_dir = param.IMG_DIR
label_dir = param.LABLE_DIR

CONTEXT_AREA = param.PATCH_SIZE  # 像素块大小
''' 读图 '''
dataset = ImageDataset(img_dir, label_dir, CONTEXT_AREA)
print(dataset.item_filenames)

''' *************************  切块  ************************* '''

outpath_tuanju = param.TUANJU_DIR
outpath_others = param.OTHER_DIR

for i in range(len(dataset)):
    num_i = dataset[i][1].shape[0] * dataset[i][1].shape[1]  # 根据标签图大小确定num,因为img被padding过
    image_name = dataset.item_filenames[i]['img'].split('.')[0].split('/')[-1]
    print('The num of image {}: '.format(image_name), num_i)
    img, label = dataset[i]
    # img和label都是被plt读入，归一化到0-1之间了
    extract_patches(outpath_tuanju, outpath_others, img, label, CONTEXT_AREA, image_name)
print('All patches numbers: ', (len(os.listdir(outpath_tuanju)) + len(os.listdir(outpath_others))))
print('tuanju num: ', len(os.listdir(outpath_tuanju)))
print('Others num: ', len(os.listdir(outpath_others)))
