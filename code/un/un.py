import os
import time
from collections import Counter

import cv2
import numpy as np
from skimage import segmentation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from tqdm import tqdm
from statistics import mean


img_dir = '/root/CV_Project/Unsupervised segmentation/data/image'

class Args(object):
    # 图像路径
    # input_image_path = '/root/CV_Project/awesome-semantic-segmentation-pytorch-master/unsupervised/image/10un_2.jpg'  # image/coral.jpg image/tiger.jpg
    img_filenames = [os.path.join(img_dir, filename) for filename in
                     sorted(os.listdir(img_dir))]  # sorted排序文件名，使得一一对应
    out_dir = '/root/CV_Project/Unsupervised segmentation/data/pred4'
    # self.label_dir = label_dir
    # label_filenames = [os.path.join(self.label_dir, filename) for filename in sorted(os.listdir(self.label_dir))]
    # assert len(img_filenames) == len(label_filenames), "The number of images must be equal to label"
    # 最大迭代次数
    train_epoch = 2 ** 6
    # 层数
    mod_dim1 = 64  #
    mod_dim2 = 32

    min_label_num = 4  # if the label number small than it, break loop 下限阈值
    max_label_num = 256  # if the label number small than it, start to show result image.

# 网络结构
class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def run(image_filename):
    start_time0 = time.time()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    image = cv2.imread(image_filename)

    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
    ''' 在原图上看下效果 '''
    mark = segmentation.mark_boundaries(image, seg_map)
    plt.imshow(mark)
    plt.show()
    plt.imsave(os.path.join('/root/CV_Project/Unsupervised segmentation/gif/F', 'P.png'), mark)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0] # 找到索引值
               for u_label in np.unique(seg_map)] # np.unique 该函数是去除数组中的重复数字，并进行排序之后输出。

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    '''train loop'''
    start_time1 = time.time()
    model.train()
    for batch_idx in tqdm(range(args.train_epoch)):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0] # (32,960,1280)
        # 展平（1228800,32）
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        # 取32个特征图中最大值作为每个像素的标签
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()
        # plt.subplot(131)
        F_mark = segmentation.mark_boundaries(image, im_target.reshape(image.shape[:2]))
        # plt.imshow(F_mark)
        plt.imsave(os.path.join('/root/CV_Project/Unsupervised segmentation/gif/F', '{}.png').format(batch_idx), F_mark)

        # plt.imsave(os.path.join('/root/CV_Project/Unsupervised segmentation/gif', '{}.png').format(batch_idx), im_target.reshape(image.shape[:2]),cmap = 'gray')

        # plt.imshow(im_target.reshape(image.shape[:2]),cmap='gray')
        # plt.show()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True) # 统计每个聚类中，出现次数最多的类别
            im_target[inds] = u_labels[np.argmax(hist)] # 将这个聚类中的所有像素，都记录为这个类别
        plt.subplot(132);
        M_mark = segmentation.mark_boundaries(image, im_target.reshape(image.shape[:2]))
        plt.imsave(os.path.join('/root/CV_Project/Unsupervised segmentation/gif/M', '{}.png').format(batch_idx), M_mark)
        # plt.imshow(M_mark)
        # plt.imshow(im_target.reshape(image.shape[:2]), cmap ='gray')
        # plt.show()
        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]  #根据im_target划分的区域结果，取原图该区域均值
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color # 均值布置到各个超像素区域
            show = img_flatten.reshape(image.shape)
        # plt.subplot(133);
        # plt.imshow(show[:,:,0])
        # plt.show()

        # print('Loss:', batch_idx, loss.item())
        if len(un_label) < args.min_label_num:
            break

    '''save'''
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    show = show[:, :, 0]
    pixels = list(Counter(show.flatten()))
    print(pixels)

    show[show <= mean(pixels)] = 0
    show[show > mean(pixels)] = 255
    plt.imsave(os.path.join(args.out_dir, 'pred_{}.png').format(image_filename.split('.')[0].split('/')[-1]), show, cmap="gray")


if __name__ == '__main__':
    args = Args()
    for image_filename in args.img_filenames:
        print('--------------'+image_filename+':')
        run(image_filename)