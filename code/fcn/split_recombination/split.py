import os
from PIL import Image
import numpy as np
img_path = '/root/Dataset/pred_images/image'
mask_path = '/root/Dataset/pred_images/label'
out_img = '/root/Dataset/640_480_pred/image'
out_mask = '/root/Dataset/640_480_pred/label'
img_filenames = [os.path.join(img_path, filename) for filename in sorted(os.listdir(img_path))] # sorted排序文件名，使得一一对应
mask_filenames = [os.path.join(mask_path, filename) for filename in sorted(os.listdir(mask_path))]
assert len(img_filenames) == len(mask_filenames), 'The number of images must be equal to masks'
item_num = len(img_filenames)  # 图片数量
item_filenames = []  # 储存字典{image:label}

for i in range(item_num):
    ''' 字典形式存储每个样本数据和label
     '''
    item_filenames.append({
        'image': img_filenames[i],
        'mask': mask_filenames[i]
    })

for i in range(item_num):
    item_name = item_filenames[i]
    base_name = os.path.basename(img_filenames[i]).split('.')[0] # 所属图片名
    img = Image.open(item_name['image']).convert('RGB')
    mask = Image.open(item_name['mask'])

    size = img.size
    # 准备将图片切割成4张小图片
    weight = int(size[0] // 2)
    height = int(size[1] // 2)

    # 切割后的小图的宽度和高度
    print(weight, height)

    for m in range(2):
        for n in range(2):
            box = (weight * n, height * m, weight * (n + 1), height * (m + 1))
            region_img = img.crop(box)
            region_mask = mask.crop(box)
            region_img.save('{}/{}_{}{}.png'.format(out_img, base_name, m, n))
            region_mask.save('{}/{}_{}{}.png'.format(out_mask, base_name, m, n))
