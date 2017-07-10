'''
crop&rescale images/labels to 1024x1024
'''

# # copy ex data to the specified folder
# import os
# import shutil
# from glob import glob
#
# dirs = os.listdir('/Users/zhangweidong03/data/ex/e_optha_EX/EX')
# for dir in dirs:
#     images = glob(os.path.join('/Users/zhangweidong03/data/ex/e_optha_EX/EX/'+dir, '*.jpg'))
#     if len(images) > 0:
#         for image in images:
#             print(image)
#             shutil.copy(image, '/Users/zhangweidong03/data/ex/e_optha_EX/images/')
#
#
#
# dirs = os.listdir('/Users/zhangweidong03/data/ex/e_optha_EX/Annotation_EX')
# for dir in dirs:
#     labels = glob(os.path.join('/Users/zhangweidong03/data/ex/e_optha_EX/Annotation_EX/' + dir, '*.png'))
#     if len(labels) > 0:
#         for label in labels:
#             print(label)
#             shutil.copy(label, '/Users/zhangweidong03/data/ex/e_optha_EX/labels/')


import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Microaneurysm data processing')
    parser.add_argument('--root', required=True, help='the root path include following path: images, labels, ...')

    return parser.parse_args()

opt = arg_parse()
print(opt)


import os

images_path = os.path.join(opt.root, 'images')
labels_path = os.path.join(opt.root, 'labels')
images_1024_path = os.path.join(opt.root, 'images_1024')
labels_1024_path = os.path.join(opt.root, 'labels_1024')

# check path exist
if not os.path.exists(images_path):
    print('Error! {} is not exist!'.format(images_path))

if not os.path.exists(labels_path):
    print('Error! {} is not exist!'.format(labels_path))

if not os.path.exists(images_1024_path):
    os.mkdir(images_1024_path)
    print('Create images path for 1024x1024 resolution:')

if not os.path.exists(labels_1024_path):
    os.mkdir(labels_1024_path)
    print('Create label path for 1024x1024 resotlution:')

from glob import glob
from PIL import Image

images_list = glob(os.path.join(images_path, '*.jpg'))
labels_list = glob(os.path.join(labels_path, '*.png'))

scale_size = 1024

for i in images_list:
    base_str = os.path.basename(i).split('.')[0]
    # l = os.path.join(labels_path, base_str+'_mask.png')
    l = os.path.join(labels_path, base_str+'_EX.png')
    if l in labels_list:
        img = Image.open(i)
        label = Image.open(l)
        w,h = img.size
        tw = th = (min(w,h))
        img = img.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
        label = label.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
        tw, th = (1024, 1024)
        w,h = img.size
        ratio = tw/w
        assert ratio == th/h
        if ratio <= 1:
            img.resize((tw, th), Image.ANTIALIAS)
            label.resize((tw, th), Image.ANTIALIAS)
        img.save(os.path.join(images_1024_path, base_str+'_1024.png'))
        label.save(os.path.join(labels_1024_path, base_str+'_label_1024.png'))
    else:
        continue

