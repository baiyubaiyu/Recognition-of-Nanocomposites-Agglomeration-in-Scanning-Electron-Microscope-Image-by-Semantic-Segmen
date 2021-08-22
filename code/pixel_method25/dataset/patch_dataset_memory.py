'''
    预测时，将预测图片切成的像素块直接放在内存里（存在list中—），没有保存下来，因此写了一个从内存读取patch的脚本
'''
import torch.utils.data as data


class Patch(data.Dataset):

    def __init__(self, patches_list, labels_list):
        assert len(patches_list) == len(labels_list), 'The number of patches must be equal to labels'
        self.patches = patches_list
        self.labels = labels_list
        self.num = len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        return patch, label

    def __len__(self):
        return self.num

if __name__ == '__main__':
    patches = [1,2,3]
    labels = ['a',  'b', 'c']
    datset = Patch(patches, labels)
    loader = data.DataLoader(datset, 2, shuffle=True)
    for data, target in loader:
        print(data)
        print(target)

