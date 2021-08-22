'''
    切图程序，并步保存文件，而是放置在内存中
'''
import numpy as np
from dataset.image_dataset import ImageDataset
from collections import Counter


def sample_all(Image, Label, context_area, pred=False):
    '''
    采样训练数据
    :param Image: padding后的图，使得可以取到原图边缘点
    :param Label: Label图
    :param context_area: 像素上下文环境大小m

    :return: patches and pixel_labels: <class 'list'>;  items: (<class 'numpy.ndarray', class 'numpy.int64'>
    '''
    if pred == False:
        assert Image.shape[1] == (Label.shape[0] + context_area-1), 'The shape of image(before padding) must be equal to label.'
        assert Image.shape[2] == (Label.shape[1] + context_area-1), 'The shape of image(before padding) must be equal to label.'
        ''' Extract patches '''

        ''' Image: channels, height, width'''
        patches = []
        pixel_labels = []
        for row in range(Label.shape[0]):
            for col in range(Label.shape[1]):
                patches.append(Image[:, row:(row + context_area), col:(col + context_area)])
                pixel_labels.append(Label[row, col])
        return patches, pixel_labels
    else:
        patches = []
        for row in range(Image.shape[1]+1-context_area):
            for col in range(Image.shape[2]+1-context_area):
                patches.append(Image[:, row:(row + context_area), col:(col + context_area)])
        return patches

if __name__ == '__main__':
    EPOCH = 10
    BATCH_SIZE = 75
    train_root_dir = "/home/baiyu/Dataset/maotai/Train_data"
    val_root_dir = "/home/baiyu/Dataset/maotai/Val_data"
    img_dir = "images"
    label_dir = 'labels'
    context_area = 25
    dataset = ImageDataset(val_root_dir, img_dir, label_dir, 25)
    image, label = dataset[0]
    patch, pixel_label = sample_all(image, label, 25)
    print(len(patch), len(pixel_label))
    lenss = []
    for p in patch:
        lenss.append(p.shape[1])
    print(Counter(lenss))






