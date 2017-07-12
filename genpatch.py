from PIL import Image
import math
import numpy as np

import os

def image_basename(filename):
    return os.path.basename(filename).split('.')[0]

def image_path(root, basename):
    path = ''
    if basename.startswith('C'):
        path = os.path.join(root, '{}.jpg'.format(basename))
    else:
        path = os.path.join(root, '{}.JPG'.format(basename))
    return path

def label_path(root, basename):
    return os.path.join(root, '{}.png'.format(basename))

# (row, column)
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


patch_size = (256,256)

# im = Image.open('/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata/labels/C0001275.png')
#
# im1 = np.array(im)
#
# patch_list = get_patch_pos(im1.shape, patch_size, im1)
#
# print(patch_list)


# filenames = [image_basename(f) for f in os.listdir('/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata/labels')]
# filenames.sort()
#
# image_root = '/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata/images'
# label_root = '/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata/labels'
#
# image_gen_root = '/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata_patches/images'
# label_gen_root = '/Users/zhangweidong03/Code/dl/pytorch/github/piwise/MAdata_patches/labels'

# for f in filenames:
#     im_labels = Image.open(label_path(label_root, f))
#     im1_labels = np.array(im_labels)
#     im_images = Image.open(image_path(image_root, f))
#     im1_images = np.array(im_images)
#     patch_list = get_patch_pos(im1_labels.shape, patch_size, im1_labels)
#     for p in patch_list:
#         iml = im1_labels[
#             p[0] * patch_size[0]:(p[0]+1) * patch_size[0], p[1] * patch_size[1]:(p[1]+1) * patch_size[1]
#         ]
#         imi = im1_images[
#               p[0] * patch_size[0]:(p[0] + 1) * patch_size[0], p[1] * patch_size[1]:(p[1] + 1) * patch_size[1]
#         ]
#         pil_iml = Image.fromarray(iml)
#         pil_imi = Image.fromarray(imi)
#         pil_iml.save(label_gen_root+'/'+'{}_{}_{}.png'.format(f, p[0], p[1]))
#         pil_imi.save(image_gen_root + '/' + '{}_{}_{}.jpg'.format(f, p[0], p[1]))


filenames = [image_basename(f) for f in os.listdir('/Users/zhangweidong03/data/ex/ex/labels')]
filenames.sort()

image_root = '/Users/zhangweidong03/data/ex/ex/images'
label_root = '/Users/zhangweidong03/data/ex/ex/labels'

image_gen_root = '/Users/zhangweidong03/data/ex/ex_patches/images'
label_gen_root = '/Users/zhangweidong03/data/ex/ex_patches/labels'


for f in filenames:
    im_labels = Image.open(label_path(label_root, f))
    im1_labels = np.array(im_labels)
    img = image_path(image_root, f.replace('_EX', ''))
    if not os.path.isfile(img):
        continue
    im_images = Image.open(image_path(image_root, f.replace('_EX', '')))
    im1_images = np.array(im_images)
    patch_list = get_patch_pos(im1_labels.shape, patch_size, im1_labels)
    for p in patch_list:
        iml = im1_labels[
            p[0] * patch_size[0]:(p[0]+1) * patch_size[0], p[1] * patch_size[1]:(p[1]+1) * patch_size[1]
        ]
        imi = im1_images[
              p[0] * patch_size[0]:(p[0] + 1) * patch_size[0], p[1] * patch_size[1]:(p[1] + 1) * patch_size[1]
        ]
        pil_iml = Image.fromarray(iml)
        pil_imi = Image.fromarray(imi)
        pil_iml.save(label_gen_root+'/'+'{}_{}_{}.png'.format(f, p[0], p[1]))
        pil_imi.save(image_gen_root + '/' + '{}_{}_{}.png'.format(f, p[0], p[1]))
