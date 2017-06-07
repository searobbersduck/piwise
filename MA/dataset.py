from PIL import Image

from torch.utils.data import Dataset

import os

import math

__all__ = [
    'MA',
    'image_basename',
    'eval_ds',
]

EXTENSIONS = ['.jpg', '.png', '.JPG']


def load_image(file):
    return Image.open(file)

def is_image(file):
    return any(file.endswith(ext) for ext in EXTENSIONS)

def image_basename(filename):
    return os.path.basename(filename).split('.')[0]

# def image_path(root, basename):
#     path = ''
#     if basename.startswith('C'):
#         path = os.path.join(root, '{}.jpg'.format(basename))
#     else:
#         path = os.path.join(root, '{}.JPG'.format(basename))
#     return path


def image_path(root, basename):
    path = ''
    path = os.path.join(root, '{}.jpg'.format(basename))
    return path


def label_path(root, basename):
    return os.path.join(root, '{}.png'.format(basename))




class MA(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        filename = self.filenames[item]

        with open(image_path(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(label_path(self.labels_root, filename), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)



class eval_ds(Dataset):
    def __init__(self, imagepath, input_transform, patch_size=(256,256)):
        self.image = Image.open(imagepath)
        self.row = math.ceil(self.image.size[1]/patch_size[0])
        self.column = math.ceil(self.image.size[0]/patch_size[1])
        # self.row = 4
        # self.column = 4
        self.patch_size = patch_size
        self.len = self.row * self.column
        self.input_transform = input_transform
        print('init')

    def __getitem__(self, item):
        idx = int(item)
        currow = idx // self.column
        curcolumn = idx % self.column
        image_patch = self.image.crop((curcolumn*self.patch_size[1],
                                       currow * self.patch_size[0],
                                       (curcolumn+1)*self.patch_size[1],
                                      (currow + 1) * self.patch_size[0])
        )
        return self.input_transform(image_patch), currow, curcolumn

    def __len__(self):
        return self.len


def main():
    ma = MA('/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata')

    for i, (image, label) in enumerate(ma):
        print('{}: {}/{}'.format(i , image, label))


if __name__ == '__main__':
    main()