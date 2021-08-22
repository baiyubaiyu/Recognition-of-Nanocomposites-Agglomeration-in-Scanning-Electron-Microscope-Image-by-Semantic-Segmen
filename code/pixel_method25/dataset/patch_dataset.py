'''
    像素块dataset
'''


import torch.utils.data as data
from dataset.read_image import readImage
# import imageio
# import imgaug as ia
# import imgaug.augmenters as iaa
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


class PatchDataset(data.Dataset):

    def __init__(self, filenames, transform = False):

        self.item_filenames = []  # 所有整图文件路径列表
        self.item_num = len(filenames)
        # self.transform = transform
        # self.trans = iaa.WithChannels(
        #         channels=[0, 1, 2],
        #         children=iaa.Sequential([
        #             iaa.GammaContrast(gamma=random.uniform(0.7, 1.3))
        #         ])
        #     )

        for i in range(self.item_num):
            ''' 字典形式存储每个样本数据和label
             '''
            self.item_filenames.append({
                'img': filenames[i],
                'label': int(filenames[i][-5])
            })

    def __len__(self):
        return self.item_num

    def __getitem__(self, idx):
        item_name = self.item_filenames[idx]
        ''' 读图 '''
        image = readImage(item_name['img'])  # 3*25*25
        image[[0, 2], :, :] = image[[2, 0], :, :] # bgr -> rgb
        # if self.transform == True:
        #     image = image.transpose(1, 2, 0)  # （H, W, C)
        #     if random.randint(0,10)>=5:
        #         image = self.trans.augment_image(image)
        #     image = image.transpose(2,0,1)
        label = item_name['label']  # int
        return image, label

if __name__ == '__main__':
    gaoliang_dir = '/home/baiyu/Data/Train_SEM/patches/others'
    # other_dir = '/home/baiyu/Dataset/maotai_3_4/patch_maxmin4/170414/other'
    gaoliang_filenames = [os.path.join(gaoliang_dir, filename) for filename in sorted(os.listdir(gaoliang_dir))[0:5]]
    dataset_train = PatchDataset(filenames=gaoliang_filenames,transform=True)
    print(gaoliang_filenames[0])
    # print(dataset_train[0][0] * 255)
    x = dataset_train[0][0]
    # x[[0,2],:,:] = x[[2,0],:,:]
    # x = np.transpose(x, (1,2,0))5
    # plt.imshow(x.astype(int))
    # plt.show()

    a = '/home/baiyu/Data/Train_SEM/image/10C3_1.jpg'
    # a =gaoliang_filenames[0]
    img = plt.imread(a)
    y = img[0:13,0:13,:]
    y = np.transpose(y, (2,0,1))

    print(img.shape)
    # print(img.shape)
    # print(img[0:13,0:13,0])
    # /home/baiyu/Data/Train_SEM/patches/tuanju/5C3_1_915938_0.png