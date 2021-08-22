'''
    图像dataset
'''

import torch.utils.data as data
import os
from dataset.read_image import readImage
from math import floor
from dataset.transform.padding import reflect_pad2d

class ImageDataset(data.Dataset):

    def __init__(self, img_dir, label_dir, context_area):
        self.img_dir = img_dir
        self.label_dir = label_dir
        img_filenames = [os.path.join(self.img_dir, filename) for filename in sorted(os.listdir(self.img_dir))] # sorted排序文件名，使得一一对应
        label_filenames = [os.path.join(self.label_dir, filename) for filename in sorted(os.listdir(self.label_dir))]
        assert len(img_filenames) == len(label_filenames), "The number of images must be equal to label"

        self.item_num = len(img_filenames)  # 图片数量
        self.item_filenames = []  # 储存字典{image:label}
        self.context_area = context_area # 块大小

        for i in range(self.item_num):
            ''' 字典形式存储每个样本数据和label
             '''
            self.item_filenames.append({
                'img': img_filenames[i],
                'label': label_filenames[i]
            })

    def __len__(self):
        return self.item_num

    def __getitem__(self, idx):
        item_name = self.item_filenames[idx]

        ''' 读图 '''
        pad_size = floor(self.context_area / 2)
        image = readImage(item_name['img'])
        pad_image = reflect_pad2d(image, pad_size) # padding: 用于边缘点切块
        label = readImage(item_name['label'])
        return pad_image.numpy(), label



#
if __name__ == '__main__':
    img_dir = "/home/baiyu/Data/Train_SEM/image"
    label_dir = "/home/baiyu/Data/Train_SEM/label"
    batch_size = 3
    train_dataset = ImageDataset(img_dir, label_dir, context_area=25)  # data.Dataset

    # 测试dataset性质和元素：
    print('\ndataset_samples_num: {0}\n'.format(len(train_dataset)))
    print(train_dataset.item_filenames)
    # print("first item: ")
    # print('type: {0},    len: {1}\n'.format(type(train_dataset[0]), len(train_dataset[0])))  # 返回元组形式，元组每个元素是
#     print(train_dataset[0])
#     #
#     # mean = [np.mean(train_dataset[0][0][0]), np.mean(train_dataset[0][0][1]), np.mean(train_dataset[0][0][2]), np.mean(train_dataset[0][0][3])]
#     # print(mean)
#     # print('data:')
#     # print(type(train_dataset[0][0]), train_dataset[0][0].dtype, train_dataset[0][0].shape)
#     # print('label')
#     # print(type(train_dataset[0][1]), train_dataset[0][1].dtype, train_dataset[0][1])
#     # print(train_dataset[0][1])
#
#
#
    # trainloader = data.DataLoader(train_dataset,
    #                               batch_size=batch_size)  # <class 'torch.utils.data.dataloader.DataLoader'>
    #
    # # 测试DataLoader性质和元素：
    # for item in trainloader:
    #     print(type(item)) # list[tensor:image, tensor:label]
    #     print(len(item))
    #     print(type(item[0]), item[0].shape) # .<class 'torch.Tensor'> torch.Size([2, 4, 250, 250])
    #     print(type(item[1]), item[1].shape) # <class 'torch.Tensor'> torch.Size([2, 250, 250])

        # print(type(item[0][0]),item[0][0].shape)
        # print(type(item[1][0]), item[1][0].shape)

        # break
