'''
1. python /Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/data/ma_data_postprocessing_stitching.py --root ./ --type cropped --metrics f1

'''

import argparse
import os
from glob import glob
from PIL import Image
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='stitching images&labels&evaluated labels into one image')
    parser.add_argument('--root', required=True)
    parser.add_argument('--type', default='none_cropped', choices=['none_cropped', 'cropped', 'maxcc'])
    parser.add_argument('--metrics', default='f1', choices=['f1', 'iou'])

    return parser.parse_args()

args = parse_args()

root = args.root

dict = {
    'none_cropped': ['stat_res_none_cropped.txt', 'stitching_images_none_cropped', 'eval_labels_none_cropped'],
    'cropped': ['stat_res_cropped.txt', 'stitching_images_cropped', 'eval_labels_cropped'],
    'maxcc': ['stat_res_maxcc.txt', 'stitching_images_maxcc', 'eval_labels_maxcc']
}

stitching_images_file = dict[args.type][1]+'_{}'.format(args.metrics)
res_file = dict[args.type][0].replace('.txt', '_{}.txt'.format(args.metrics))

root_eval_labels_postprocessing = os.path.join(args.root, '{0}_{1}'.format(dict[args.type][2], args.metrics))

images_path = os.path.join(root, 'images')
labels_path = os.path.join(root, 'labels')
eval_labels_path = root_eval_labels_postprocessing
stitching_images_path = os.path.join(root, stitching_images_file)

if not os.path.isdir(stitching_images_path):
    os.mkdir(stitching_images_path)

assert os.path.isdir(stitching_images_path)

images_list = glob(os.path.join(images_path, "*.png"))
labels_list = glob(os.path.join(labels_path, '*.png'))
eval_labels_list = glob(os.path.join(eval_labels_path, '*.png'))

images_list = [os.path.basename(index) for index in images_list]
labels_list = [os.path.basename(index) for index in labels_list]
eval_labels_list = [os.path.basename(index) for index in eval_labels_list]


for index in images_list:
    base_str = os.path.basename(index).split('.')[0]
    label_index = base_str+'_mask.png'
    eval_label_index = base_str + '_{}.png'.format('eval_label')
    if not label_index in labels_list or not eval_label_index in eval_labels_list:
        continue
    stitching_index = base_str + '_stitching.png'
    stitching_index = os.path.join(stitching_images_path, stitching_index)
    image = Image.open(os.path.join(images_path, index))
    label = Image.open(os.path.join(labels_path, label_index))
    eval_label = Image.open(os.path.join(eval_labels_path, eval_label_index))

    assert image.size == label.size == eval_label.size

    image_stitching = Image.new('RGB', (image.size[0]*3, image.size[1]))

    image_stitching.paste(image, (0, 0))
    image_stitching.paste(label, (1*image.size[0], 0))
    image_stitching.paste(eval_label, (2*image.size[0], 0))

    print('====>save: {}'.format(stitching_index))
    image_stitching.save(stitching_index)


cat_range = [
    [0.0, 0.25],
    [0.25, 0.5],
    [0.5, 0.75],
    [0.75, 1.0]
]

config_file = os.path.join(root, res_file)

cat_path = os.path.join(root, 'cata')
if not os.path.isdir(cat_path):
    os.mkdir(cat_path)
assert os.path.isdir(cat_path)

cat_path = os.path.join(cat_path, stitching_images_file)
if not os.path.isdir(cat_path):
    os.mkdir(cat_path)
assert os.path.isdir(cat_path)

for range in cat_range:
    cat_path = os.path.join(root, 'cata/{2}/{0}-{1}'.format(range[0], range[1], stitching_images_file))
    if not os.path.isdir(cat_path):
        os.mkdir(cat_path)
    if not os.path.isdir(cat_path):
        continue

with open(config_file) as f:
    lines = f.readlines()
    for line in lines:
        if len(line) > 10:
            words = line.strip().split('\t')
            index = words[0]
            sensitivity = float(words[3].split(':')[1])
            index = index.replace('.png', '_stitching.png')
            index = os.path.join(stitching_images_path, index)
            for range in cat_range:
                cat_path = os.path.join(root, 'cata/{2}/{0}-{1}'.format(range[0], range[1], stitching_images_file))
                if sensitivity >= range[0] and sensitivity <= range[1]:
                    shutil.copy(index, cat_path)


