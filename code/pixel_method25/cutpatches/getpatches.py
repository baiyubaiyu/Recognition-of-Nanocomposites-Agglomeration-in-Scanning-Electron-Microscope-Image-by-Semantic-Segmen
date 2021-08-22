'''
    切图函数
'''
from tqdm import tqdm
import cv2
import numpy as np
import os
def extract_patches(outpath_tuanju, outpath_others, image, label, context_area, image_name):
    '''
    将图像切为context_area * context_area的patch，并保存到指定路径
    :param outpath_tuanju: tuanju保存路径
    :param outpath_others: 非团聚保存路径
    :param image: 原始图像（这里是经过context_area镜像padding的image）
    :param label: 原图label
    :param context_area: patch大小
    :param image_name: 图像标识名，方便给patch命名
    :return:
    '''

    assert image.shape[1] == (label.shape[0] + context_area - 1), 'The shape of image(before padding) must be equal to label.'
    assert image.shape[2] == (label.shape[1] + context_area - 1), 'The shape of image(before padding) must be equal to label.'

    idx = 0
    ''' Extract patches '''
    for row in tqdm(range(label.shape[0])):
        for col in range(label.shape[1]):
            patch = image[:, row:(row + context_area), col:(col + context_area)]
            pixel_label = label[row, col]
            if pixel_label == 0:
                filename = outpath_tuanju + '/{}_{:d}_0.png'.format(image_name, idx)
                # if filename in os.listdir(outpath_tuanju):
                #     break
                cv2.imwrite(filename, np.transpose(patch,(1,2,0)))
            else:
                filename = outpath_others + '/{}_{:d}_1.png'.format(image_name, idx)
                # if filename in os.listdir(outpath_tuanju):
                #     break
                cv2.imwrite(filename, np.transpose(patch,(1,2,0)))   # 注意这里用cv2.imwrite, 会将图片保存成BGR
            idx += 1
                # break
        # break


