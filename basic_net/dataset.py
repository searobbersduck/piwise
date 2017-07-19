from PIL import Image

from torch.utils.data import Dataset

import os

__all__ = [
    'dt_ma'
]

class dt_ma(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'ahe_images')
        self.labels_root = os.path.join(root, 'labels')

        self.base_str_filenames = [os.path.basename(f).split('.')[0] for f in os.listdir(self.images_root)]
        self.base_str_filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform


    def __getitem__(self, item):
        filename = self.base_str_filenames[item]
        with open(os.path.join(self.images_root, filename+'.png'), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(os.path.join(self.labels_root, filename.replace('_ahe', '_mask')+'.png'), 'rb') as f:
            label = Image.open(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.base_str_filenames)


class dt_ex(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None, resolution=512):
        self.images_root = os.path.join(root, 'images_{}'.format(resolution))
        self.labels_root = os.path.join(root, 'labels_{}'.format(resolution))
        self.resolution = resolution

        self.base_str_filenames = [os.path.basename(f).split('.')[0].split('_')[0] for f in os.listdir(self.images_root)]
        self.base_str_filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        filename = self.base_str_filenames[item]
        with open(os.path.join(self.images_root, filename+'_{}.png'.format(self.resolution)), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(os.path.join(self.labels_root, filename+'_label_{}.png'.format(self.resolution)), 'rb') as f:
            label = Image.open(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.base_str_filenames)


class eval_dt_ma(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'ahe_images')

        self.base_str_filenames = [os.path.basename(f).split('.')[0] for f in os.listdir(self.images_root)]
        self.base_str_filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform


    def __getitem__(self, item):
        filename = self.base_str_filenames[item]
        with open(os.path.join(self.images_root, filename+'.png'), 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.input_transform is not None:
            image = self.input_transform(image)

        return image, filename

    def __len__(self):
        return len(self.base_str_filenames)