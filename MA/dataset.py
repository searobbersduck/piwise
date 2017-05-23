from PIL import Image

from torch.utils.data import Dataset

import os

__all__ = [
    'MA',
    'image_basename'
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

def main():
    ma = MA('/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata')

    for i, (image, label) in enumerate(ma):
        print('{}: {}/{}'.format(i , image, label))


if __name__ == '__main__':
    main()