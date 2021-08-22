import os
import numpy as np
import matplotlib.pyplot as plt

def readImage(filename):
    '''
    读取.img格式遥感图像
    :param filename: 文件名
    :return: image <class 'numpy.ndarray'>
    '''
    assert os.path.exists(filename), "No such directory"

    image = plt.imread(filename)
    ''' 区分data和label '''
    if len(image.shape) > 2: #image
        image = np.transpose(image, (2, 0, 1))  # pytorch框架下rensor中通道顺序（batch_size, channel, height, width)
        image = image.astype(np.float32)
    return image

def pixel_hist(image):
    '''
    绘制图像像素直方图
    image:(c,h,w)图像
    label:(h,w)
    '''
    if len(image.shape) > 2: # image (3 x 960 x 1280)
        fig, axs = plt.subplots(image.shape[2], 1)
        for i in range(image.shape[2]):
            num_dict = image[:, :,i].flatten()
            axs[i].hist(num_dict, bins='auto')
        plt.show()
    else:  # label (960 x 1280)
        fig, axs = plt.subplots()
        num_dict = image.flatten()
        axs.hist(num_dict, bins='auto')
        plt.show()


''' 验证程序 '''
if __name__ == '__main__':

    ''' 像素直方图 '''
    filename = '/home/baiyu/Data/Train_SEM/image/10C3_1.jpg'
    # print(os.listdir(filename))
    image = readImage(filename)
    print(image.shape)
    image = np.transpose(image,(1,2,0))
    # print(image.shape)
    # image = np.transpose(image, (2,0,1))
    # print(image)
    # print(type(image[0][0][0]))
    plt.imshow(image.astype(int))
    # plt.imshow(image)
    plt.show()
    image = np.transpose(image, (2,0,1))
    pixel_hist(image)

    # import os
    #
    # outpath_tuanju = '/home/baiyu/Data/SEM/patches/tuanju'
    # outpath_others = '/home/baiyu/Data/SEM/patches/others'
    # print('baiyu')
    # print(len(os.listdir(outpath_tuanju)))

