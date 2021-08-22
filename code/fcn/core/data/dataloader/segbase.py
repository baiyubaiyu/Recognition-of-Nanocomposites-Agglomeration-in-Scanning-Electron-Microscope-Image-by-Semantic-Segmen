"""Base segmentation dataset"""
import random
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image, ImageOps, ImageFilter

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=900):
        '''
            root: 路径
            split： 进一步路径
            mode： train-val-test
            transform: transform
            base_size: 设定的基础尺寸，用于尺度变化
            crop_size: 随机裁剪的大小
        '''
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        # outsize = self.crop_size
        # short_size = outsize
        # w, h = image.size
        # if w > h:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # else:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        # image = image.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # # center crop
        # w, h = image.size
        # x1 = int(round((w - outsize) / 2.))
        # y1 = int(round((h - outsize) / 2.))
        # image = image.crop((x1, y1, x1 + outsize, y1 + outsize))
        # mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        '''
            训练数据增强
        '''
        img, mask = self._img_transform(img), self._mask_transform(mask) #转numpy

        image = np.expand_dims(img, 0) # 扩充成4维，满足函数调用条件
        mask = np.expand_dims(mask, 0)
        mask = np.expand_dims(mask, 3)
        seq = iaa.Sequential([
            # iaa.Affine(rotate=(-25, 25)), # 旋转-25,25度
            iaa.Crop(percent=(0, 0.2)), # 随机裁剪
            iaa.Fliplr(0.5), # 左右镜像
            iaa.Flipud(0.5), # 上下镜像
            # iaa.AdditiveGaussianNoise(scale=(10, 60)),
            iaa.Rot90([1, 3]), # 90度旋转
        ])
        images_aug, mask_aug = seq(images=image, segmentation_maps=mask)
        images_aug = np.squeeze(images_aug) # 删除冗余的维度
        mask_aug = np.squeeze(mask_aug)
        # ia.imshow(images_aug)
        # ia.imshow(mask_aug)
        # random mirror 镜像，即水平翻转
        # if random.random() < 0.5:
        #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # crop_size = self.crop_size

        # # random scale (short edge) 尺度变化，宽高仍按照原比例
        # short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))  # 返回两者间的随机整数，确定变化后短边的大小
        # w, h = image.size
        # if h > w:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        # else:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # image = image.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        #
        # # pad crop 填充，随机尺度变化后又将图片填充为crop大小
        # if short_size < crop_size:
        #     padh = crop_size - oh if oh < crop_size else 0
        #     padw = crop_size - ow if ow < crop_size else 0
        #     image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
        #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        #
        # # random crop crop_size   裁剪  x1,y1为裁剪的左上角坐标， crop_size为裁剪大小
        # w, h = image.size
        # x1 = random.randint(0, w - crop_size)
        # y1 = random.randint(0, h - crop_size)
        # image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        return images_aug, mask_aug

    def _img_transform(self, img):
        return np.array(img).astype('uint8')

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
