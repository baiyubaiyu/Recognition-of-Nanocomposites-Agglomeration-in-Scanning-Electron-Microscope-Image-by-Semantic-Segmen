'''
图像增强效果展示
'''
import numpy as np
from imgaug import augmenters as iaa
import imageio
import imgaug as ia


'''
1\标签改一下
2、
'''
image = imageio.imread('/root/Data/SEM/640_480/train/images/10C3_0_00.png')
mask = imageio.imread("/root/Data/SEM/640_480/train/masks/10C3_0_00.png")


# print("Original:")
# ia.imshow(image)
# ia.imshow(mask)
# ia.seed(6)

seq = iaa.Sequential([
    # iaa.Affine(rotate=(-25, 25)), # -25,25度旋转
    # iaa.Crop(percent=(0, 0.2)), # 原图截图子图，左右上下裁剪宽度范围是20%
    # iaa.Fliplr(1),
    # iaa.Flipud(1),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    # iaa.Rot90([1,3]), # 90度旋转

])
image = np.expand_dims(image,0)
mask = np.expand_dims(mask,0)
mask = np.expand_dims(mask,3)
images_aug, mask_aug = seq(images=image,segmentation_maps=mask)
images_aug = np.squeeze(images_aug)
mask_aug = np.squeeze(mask_aug)

print("Augmented:")
ia.imshow(images_aug)
imageio.imwrite('/root/CV_Project/awesome-semantic-segmentation-pytorch-master/augimages/'+'AdditiveGaussianNoise'+'.png', images_aug)
# ia.imshow(mask_aug)
#
# images = [image, image, image, image]
# images_aug = rotate(images=images)
#
# print("Augmented batch:")
# ia.imshow(np.hstack(images_aug))