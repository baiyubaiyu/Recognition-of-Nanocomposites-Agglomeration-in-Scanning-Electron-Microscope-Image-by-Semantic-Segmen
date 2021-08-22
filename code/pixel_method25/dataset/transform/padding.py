'''
    镜像padding
'''

import torch as t
import torch.nn.functional as F
# from dataset.ReadImage import readImage

def reflect_pad2d(image, pad_size):
    '''
    镜像padding
    :param image: 图像
    :param pad_size: padding大小，都是正方形，四个方向全padding

    :return: padding后的图像 <class 'torch.Tensor'>
    '''
    original_shape = image.shape
    image = image.reshape([1]+list(original_shape)) # reshape成4D

    ''' 利用torch.nn.functional.pad() 进行padding，由于其只能padding 4D图的后两维，3D的最后一维 ，所以这里进行了两次reshape'''
    image = t.from_numpy(image)
    p1d = (pad_size, pad_size, pad_size, pad_size) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    pad_image = F.pad(image, p1d, mode='reflect')
    pad_image = pad_image.reshape(original_shape[0], original_shape[1]+2*pad_size, original_shape[2]+2*pad_size) # reshape 回3D

    return pad_image

# if __name__ == '__main__':
#     image = readImage("/home/baiyu/Dataset/maotai/Train_data/images_10c/satellite_dice_p1_1_1.img")
#     print(type(image), image.shape)
#
#     pad_image = ReflectPad2D(image, 2)
#     print(type(pad_image), pad_image.shape)