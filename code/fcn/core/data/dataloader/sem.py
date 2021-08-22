import os
import torch
# from collections import Counter
from PIL import Image
from core.data.dataloader.segbase import SegmentationDataset

class SEMSegmentation(SegmentationDataset):
    NUM_CLASS = 2

    def __init__(self, root='/root/Dataset/train_val', split='train', mode=None, transform=None, **kwargs):
        super(SEMSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        self.root = os.path.join(root, split)
        self.img_dir = os.path.join(self.root, 'images')
        self.mask_dir = os.path.join(self.root, 'masks')

        # print('train set')
        img_filenames = [os.path.join(self.img_dir, filename) for filename in
                         sorted(os.listdir(self.img_dir))]  # sorted排序文件名，使得一一对应
        mask_filenames = [os.path.join(self.mask_dir, filename) for filename in
                           sorted(os.listdir(self.mask_dir))]

        assert len(img_filenames) == len(mask_filenames), "The number of images must be equal to masks"

        self.item_num = len(img_filenames)  # 图片数量
        self.item_filenames = []  # 储存字典{image:label}

        for i in range(self.item_num):
            ''' 字典形式存储每个样本数据和label
             '''
            self.item_filenames.append({
                'image': img_filenames[i],
                'mask': mask_filenames[i]
            })

    def __len__(self):
        return self.item_num

    def __getitem__(self, index):
        item_name = self.item_filenames[index]

        img = Image.open(item_name['image']).convert('RGB')
        mask = Image.open(item_name['mask'])
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        # elif self.mode == 'pred':
        #     img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = torch.from_numpy(mask).long()
        return img, mask, os.path.basename(item_name['image']) #os.path.basename返回路径的最后一个值，即'/Users/beazley/Data/data.csv'中的data.csv

    # def _mask_transform(self, mask):
    #     target = np.array(mask).astype('int32')
    #     return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('agglomeration', 'basic')

if __name__ == '__main__':
    A = SEMSegmentation()
    import matplotlib.pyplot as plt
    x = A[0]
    plt.imshow(x[0])
    plt.show()


