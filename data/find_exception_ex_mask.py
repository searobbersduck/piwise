import argparse
from glob import glob
import os
import cv2
from PIL import Image
import numpy as np
import shutil

exception_mask_list = []

def parse_args():
    parser = argparse.ArgumentParser(description='segmentation post processing')
    parser.add_argument('--root', required=True)
    return parser.parse_args()


args = parse_args()

root = args.root

labels_path = os.path.join(root, 'labels')

assert os.path.isdir(labels_path)

labels_list = glob(os.path.join(labels_path, '*.png'))


def getMaxRegion(contour):
    rect = []
    for c in contour:
        rect.append(cv2.boundingRect(c))

    maxrect = rect[0]
    maxwxh = maxrect[2]*maxrect[3]

    for r in rect:
        if r[2]*r[3] > maxwxh:
            maxrect = r
            maxwxh = r[2]*r[3]
    return (maxrect[0], maxrect[1], maxrect[0]+maxrect[2], maxrect[1] + maxrect[3])

def getMaxRegionArea(contour):
    rect = []
    for c in contour:
        rect.append(cv2.boundingRect(c))

    maxrect = rect[0]
    maxwxh = maxrect[2]*maxrect[3]

    for r in rect:
        if r[2]*r[3] > maxwxh:
            maxrect = r
            maxwxh = r[2]*r[3]
    # return (maxrect[0], maxrect[1], maxrect[0]+maxrect[2], maxrect[1] + maxrect[3])
    return maxrect[2]*maxrect[3]

def extract_except_mask(label_index, area_thresh):
    cv_label = cv2.imread(label_index, cv2.IMREAD_GRAYSCALE)
    thresh, cv_label = cv2.threshold(cv_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(cv_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        rectArea = getMaxRegionArea(contours)
        if rectArea > area_thresh:
            return True
        else:
            return False
    return True


for label_index in labels_list:
    exception = extract_except_mask(label_index, 20000)
    if exception:
        exception_mask_list.append(label_index)

print(exception_mask_list)
print(len(exception_mask_list))

tmp_dir = os.path.join(root, 'tmp/exception')
src_dir = os.path.join(root, 'stitching_images_none_cropped_iou')

assert os.path.isdir(src_dir)

if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)

assert os.path.isdir(tmp_dir)

cat_range = [
    [0.0, 0.25],
    [0.25, 0.5],
    [0.5, 0.75],
    [0.75, 1.0]
]

for range in cat_range:
    cat_path = os.path.join(root, 'tmp/{0}-{1}'.format(range[0], range[1]))
    if not os.path.isdir(cat_path):
        os.makedirs(cat_path)
    if not os.path.isdir(cat_path):
        continue

src_files = glob(os.path.join(src_dir, '*.png'))

for index in exception_mask_list:
    index = index.replace('mask', 'stitching').replace('labels', 'stitching_images_none_cropped_iou')
    if index in src_files:
        shutil.copy(index, tmp_dir)

config_file = os.path.join(root, 'stat_res_none_cropped_iou.txt')

exception_mask_list = [os.path.basename(f).replace('_mask', '') for f in exception_mask_list]
stitching_images_path = src_dir


with open(config_file) as f:
    lines = f.readlines()
    for line in lines:
        if len(line) > 10:
            words = line.strip().split('\t')
            index = words[0]
            if index in exception_mask_list:
                continue
            sensitivity = float(words[3].split(':')[1])
            index = index.replace('.png', '_stitching.png')
            index = os.path.join(stitching_images_path, index)
            for range in cat_range:
                cat_path = os.path.join(root, 'tmp/{0}-{1}'.format(range[0], range[1]))
                if sensitivity >= range[0] and sensitivity <= range[1]:
                    shutil.copy(index, cat_path)

