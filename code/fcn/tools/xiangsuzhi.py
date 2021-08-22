'''
像素值翻转，黑白
'''
import imageio
import os

root_path = '/root/Data/SEM/640_480/train/masks'
out = '/root/Data/SEM/640_480/test/masks'
masks = [os.path.join(root_path, name) for name in os.listdir(root_path)]
for mask in masks:
    target = imageio.imread(mask)
    target[target == 0] = 2
    target[target == 255] = 0
    target[target == 2] = 1
    imageio.imwrite(os.path.join(out, os.path.basename(mask)), target)