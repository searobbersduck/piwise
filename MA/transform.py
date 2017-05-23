import numpy as np
import torch
import math

from PIL import Image

__all__ = [
    'Relabel',
    'ToLabel',
    'get_patch_pos'
]

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor > 10] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


def get_patch_pos(im_size, patch_size, label_im, thres = 127):
    list = []
    patch_row = math.ceil(im_size[0]/patch_size[0])
    patch_col = math.ceil(im_size[1] / patch_size[1])
    for i in range(patch_row):
        for j in range(patch_col):
            patch = label_im[
                i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[0]:(j+1)*patch_size[0]
            ]
            if patch.max() > thres:
                list.append((i,j))
    return list