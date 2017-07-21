'''
1. python /Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/data/ma_data_postprocessing.py --root ./ --bboxfile ./RegionInfo_id_2.txt --type cropped --metrics f1
2.
'''


import argparse
from glob import glob
import os
import cv2
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='segmentation post processing')
    parser.add_argument('--root', required=True)
    parser.add_argument('--bboxfile', default=None)
    parser.add_argument('--type', default='none_cropped', choices=['none_cropped', 'cropped', 'maxcc'])
    parser.add_argument('--metrics', default='f1', choices=['f1', 'iou'])

    return parser.parse_args()


args = parse_args()


dict = {
    'none_cropped': ['stat_res_none_cropped.txt', 'stitching_images_none_cropped', 'eval_labels_none_cropped'],
    'cropped': ['stat_res_cropped.txt', 'stitching_images_cropped', 'eval_labels_cropped'],
    'maxcc': ['stat_res_maxcc.txt', 'stitching_images_maxcc', 'eval_labels_maxcc']
}

res_file = dict[args.type][0].replace('.txt', '_{}.txt'.format(args.metrics))

assert res_file is not None

root_images = os.path.join(args.root, 'images')
root_labels = os.path.join(args.root, 'labels')
root_eval_labels = os.path.join(args.root, 'eval_labels')

assert os.path.isdir(root_images)
assert os.path.isdir(root_labels)
assert os.path.isdir(root_eval_labels)



root_eval_labels_postprocessing = os.path.join(args.root, '{0}_{1}'.format(dict[args.type][2], args.metrics))

if not os.path.isdir(root_eval_labels_postprocessing):
    os.mkdir(root_eval_labels_postprocessing)

assert os.path.isdir(root_eval_labels_postprocessing)

# root_eval_labels_cropped = os.path.join(args.root, 'eval_labels_cropped')
# if not os.path.isdir(root_eval_labels_cropped):
#     os.mkdir(root_eval_labels_cropped)
#
# assert os.path.isdir(root_eval_labels_cropped)
#
# root_eval_labels_none_cropped = os.path.join(args.root, 'eval_labels_none_cropped')
# if not os.path.isdir(root_eval_labels_none_cropped):
#     os.mkdir(root_eval_labels_none_cropped)
#
# assert os.path.isdir(root_eval_labels_none_cropped)
#
# root_eval_labels_maxcc = os.path.join(args.root, 'eval_labels_maxcc')
# if not os.path.isdir(root_eval_labels_maxcc):
#     os.mkdir(root_eval_labels_maxcc)
#
# assert os.path.isdir(root_eval_labels_maxcc)


def labels_256_postprocessing_placeholder(eval_label_path, gt_mask_list, bbox_list):
    maskpath = os.path.basename(eval_label_path).replace('_ahe_eval_label', '_mask')
    maskpath = os.path.join(root_labels, maskpath)
    bbox_index = os.path.basename(eval_label_path).replace('_ahe_eval_label', '')
    # outlabelpath = os.path.basename(eval_label_path).replace('ahe_eval_label', 'eval_label_none_cropped')
    outlabelpath = os.path.basename(eval_label_path).replace('ahe_eval_label', 'eval_label')
    pil_label_resized = None
    pil_mask = None
    if maskpath in gt_mask_list:
        cv_label = cv2.imread(eval_label_path, cv2.IMREAD_GRAYSCALE)
        thresh, cv_label = cv2.threshold(cv_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_label = Image.fromarray(cv_label).convert('L')
        pil_mask = Image.open(maskpath).convert('L')
        pil_label_resized = pil_label.resize(pil_mask.size)

        pil_label_resized.save(os.path.join(root_eval_labels_postprocessing, outlabelpath))

    return pil_label_resized, pil_mask, bbox_index

def labels_256_postprocessing_cropped(eval_label_path, gt_mask_list, bbox_list):
    maskpath = os.path.basename(eval_label_path).replace('ahe_eval_label', 'mask')
    maskpath = os.path.join(root_labels, maskpath)
    bbox_index = os.path.basename(eval_label_path).replace('_ahe_eval_label', '')
    # outlabelpath = os.path.basename(eval_label_path).replace('ahe_eval_label', 'eval_label_cropped')
    outlabelpath = os.path.basename(eval_label_path).replace('ahe_eval_label', 'eval_label')
    pil_label_cropped = None
    pil_mask_cropped = None
    if maskpath in gt_mask_list:
        cv_label = cv2.imread(eval_label_path, cv2.IMREAD_GRAYSCALE)
        thresh, cv_label = cv2.threshold(cv_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_label = Image.fromarray(cv_label)
        pil_mask = Image.open(maskpath).convert('L')
        pil_label_resized = pil_label.resize(pil_mask.size)
        pil_label_cropped = Image.new('L', pil_label_resized.size)
        crop_img = pil_label_resized.crop(bbox_list[bbox_index])
        pil_label_cropped.paste(crop_img, bbox_list[bbox_index])
        pil_label_cropped.save(os.path.join(root_eval_labels_postprocessing, outlabelpath))
        pil_mask_cropped = Image.new('L', pil_mask.size)
        crop_mask = pil_mask.crop(bbox_list[bbox_index])
        pil_mask_cropped.paste(crop_mask, bbox_list[bbox_index])
    return pil_label_cropped, pil_mask_cropped, bbox_index


# def labels_256_postprocessing_with_erodeanddilate(label_256_path, gt_mask_list, bbox_list):
#     maskpath = os.path.basename(label_256_path).replace('ahe_label', 'mask')
#     maskpath = os.path.join(args.mask, maskpath)
#     bbox_index = os.path.basename(label_256_path).replace('_ahe_label', '')
#     outlabelpath = os.path.basename(label_256_path).replace('ahe_label', 'n_label1')
#     if maskpath in gt_mask_list:
#         cv_label = cv2.imread(label_256_path, cv2.IMREAD_GRAYSCALE)
#         thresh, cv_label = cv2.threshold(cv_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         kernel = np.ones((4, 4), np.uint8)  # 生成一个6x6的核
#         erosion = cv2.erode(cv_label, kernel, iterations=1)  # 调用腐蚀算法
#         dilation = cv2.dilate(erosion, kernel, iterations=1)  # 调用膨胀算法
#         pil_label = Image.fromarray(dilation)
#         pil_mask = Image.open(maskpath).convert('L')
#         pil_label_resized = pil_label.resize(pil_mask.size)
#         pil_label_cropped = Image.new('L', pil_label_resized.size)
#         crop_img = pil_label_resized.crop(bbox_list[bbox_index])
#         pil_label_cropped.paste(crop_img, bbox_list[bbox_index])
#         pil_label_cropped.save(os.path.join(args.labelsout1, outlabelpath))
#     return pil_label_cropped, pil_mask, bbox_index
#
#
# def labels_256_postprocessing_with_erodeanddilate_cropped(label_256_path, gt_mask_list, bbox_list):
#     maskpath = os.path.basename(label_256_path).replace('ahe_label', 'mask')
#     maskpath = os.path.join(args.mask, maskpath)
#     bbox_index = os.path.basename(label_256_path).replace('_ahe_label', '')
#     outlabelpath = os.path.basename(label_256_path).replace('ahe_label', 'n_label1')
#     if maskpath in gt_mask_list:
#         cv_label = cv2.imread(label_256_path, cv2.IMREAD_GRAYSCALE)
#         thresh, cv_label = cv2.threshold(cv_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         kernel = np.ones((4, 4), np.uint8)  # 生成一个6x6的核
#         erosion = cv2.erode(cv_label, kernel, iterations=1)  # 调用腐蚀算法
#         dilation = cv2.dilate(erosion, kernel, iterations=1)  # 调用膨胀算法
#         pil_label = Image.fromarray(dilation)
#         pil_mask = Image.open(maskpath).convert('L')
#         pil_label_resized = pil_label.resize(pil_mask.size)
#         pil_label_cropped = Image.new('L', pil_label_resized.size)
#         crop_img = pil_label_resized.crop(bbox_list[bbox_index])
#         pil_label_cropped.paste(crop_img, bbox_list[bbox_index])
#         pil_label_cropped.save(os.path.join(args.labelsout1, outlabelpath))
#         pil_mask_cropped = Image.new('L', pil_mask.size)
#         crop_mask = pil_mask.crop(bbox_list[bbox_index])
#         pil_mask_cropped.paste(crop_mask, bbox_list[bbox_index])
#     return pil_label_cropped, pil_mask_cropped, bbox_index

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

def labels_256_postprocessing_with_maxconnectedcomponent_cropped(eval_label_path, gt_mask_list, bbox_list):
    maskpath = os.path.basename(eval_label_path).replace('ahe_eval_label', 'mask')
    maskpath = os.path.join(root_labels, maskpath)
    bbox_index = os.path.basename(eval_label_path).replace('_ahe_eval_label', '')
    # outlabelpath = os.path.basename(label_256_path).replace('ahe_eval_label', 'eval_label_maxcc')
    outlabelpath = os.path.basename(eval_label_path).replace('ahe_eval_label', 'eval_label')
    pil_contour_label = None
    pil_mask_cropped = None
    if maskpath in gt_mask_list:
        cv_label = cv2.imread(eval_label_path, cv2.IMREAD_GRAYSCALE)
        thresh, cv_label = cv2.threshold(cv_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((4, 4), np.uint8)  # 生成一个6x6的核
        erosion = cv2.erode(cv_label, kernel, iterations=1)  # 调用腐蚀算法
        dilation = cv2.dilate(erosion, kernel, iterations=1)  # 调用膨胀算法
        pil_label = Image.fromarray(dilation)
        pil_mask = Image.open(maskpath).convert('L')
        pil_label_resized = pil_label.resize(pil_mask.size)
        pil_label_cropped = Image.new('L', pil_label_resized.size)
        crop_img = pil_label_resized.crop(bbox_list[bbox_index])
        pil_label_cropped.paste(crop_img, bbox_list[bbox_index])

        tmp = np.array(pil_label_cropped)
        _, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            rect = getMaxRegion(contours)
            crop_contour = pil_label_resized.crop(rect)
            pil_contour_label = Image.new('L', pil_label_resized.size)
            pil_contour_label.paste(crop_contour, rect)
        else:
            pil_contour_label = pil_label_cropped

        pil_contour_label.save(os.path.join(root_eval_labels_postprocessing, outlabelpath))
        # pil_label_cropped.save(os.path.join(args.labelsout_maxcc, outlabelpath))
        pil_mask_cropped = Image.new('L', pil_mask.size)
        crop_mask = pil_mask.crop(bbox_list[bbox_index])
        pil_mask_cropped.paste(crop_mask, bbox_list[bbox_index])
    return pil_contour_label, pil_mask_cropped, bbox_index

def readBboxFile(filename):
    dict = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split(' ')
            dict[words[0]] = (int(words[1]), int(words[2]), int(words[1]) + int(words[3]), int(words[2]) + int(words[4]))
    print('bounding box pictures size: {}'.format(len(dict)))
    return dict


def metric_cal(pil_label, pil_gt_mask):
    np_labels = np.array(pil_label)
    np_gt_mask = np.array(pil_gt_mask)


    tp_cnt = 0
    fn_cnt = 0

    fp_cnt = 0
    tn_cnt = 0

    for i in range(np_labels.shape[0]):
        for j in range(np_labels.shape[1]):
            if np_gt_mask[i][j] == 255:
                if np_labels[i][j] == 255:
                    tp_cnt += 1
                else:
                    fn_cnt += 1
            else:
                if np_labels[i][j] == 255:
                    fp_cnt += 1
                else:
                    tn_cnt += 1


    sensitivity = 0
    if (tp_cnt+fn_cnt) == 0:
        sensitivity = -1
    else:
        sensitivity = tpr = tp_cnt / (tp_cnt + fn_cnt)

    specificity = tnr = tn_cnt/(tn_cnt+fp_cnt)

    f1 = 0
    if (2*tp_cnt+fp_cnt+fn_cnt) == 0:
        f1 = -1
    else:
        f1 = 2*tp_cnt/(2*tp_cnt+fp_cnt+fn_cnt)

    # return sensitivity, specificity
    return f1, specificity


def metric_cal_thres(pil_label, pil_gt_mask):
    np_labels = np.array(pil_label)
    np_gt_mask = np.array(pil_gt_mask)


    tp_cnt = 0
    fn_cnt = 0

    fp_cnt = 0
    tn_cnt = 0

    for i in range(np_labels.shape[0]):
        for j in range(np_labels.shape[1]):
            if np_gt_mask[i][j] > 127:
                if np_labels[i][j] > 127:
                    tp_cnt += 1
                else:
                    fn_cnt += 1
            else:
                if np_labels[i][j] > 127:
                    fp_cnt += 1
                else:
                    tn_cnt += 1


    sensitivity = 0
    if (tp_cnt+fn_cnt) == 0:
        sensitivity = -1
    else:
        sensitivity = tpr = tp_cnt / (tp_cnt + fn_cnt)

    specificity = tnr = tn_cnt/(tn_cnt+fp_cnt)

    return sensitivity, specificity


def metric_iou(pil_label, pil_gt_mask):
    np_labels = np.array(pil_label)
    np_gt_mask = np.array(pil_gt_mask)
    i_cnt = 0
    u_cnt = 0

    for i in range(np_gt_mask.shape[0]):
        for j in range(np_gt_mask.shape[1]):
            if np_gt_mask[i][j] > 127:
                if np_labels[i][j] > 127:
                    i_cnt += 1
                    u_cnt += 1
                else:
                    u_cnt += 1
            else:
                if np_labels[i][j] > 127:
                    u_cnt += 1

    iou = 0
    if u_cnt == 0:
        iou = -1
    else:
        iou = i_cnt / u_cnt

    return iou


args = parse_args()


dict = None

if args.bboxfile is None:
    dict = None
else:
    dict = readBboxFile(args.bboxfile)



raw_img_list = glob(os.path.join(root_images, '*.png'))
gt_mask_list = glob(os.path.join(root_labels, '*.png'))
labels_256_list = glob(os.path.join(root_eval_labels, '*.png'))

out_log_path = os.path.join(args.root, res_file)

logger = []


post_processing_strategy = None

if args.bboxfile is None:
    post_processing_strategy = labels_256_postprocessing_placeholder
else:
    if args.type == 'cropped':
        post_processing_strategy = labels_256_postprocessing_cropped
    elif args.type == 'none_cropped':
        post_processing_strategy = labels_256_postprocessing_placeholder
    elif args.type == 'maxcc':
        post_processing_strategy = labels_256_postprocessing_with_maxconnectedcomponent_cropped


def metric_f1_strategy(pil_label, pil_mask, raw_img_path):
    log = ''
    if pil_label is not None:
        sensitivity, specificity = metric_cal(pil_label, pil_mask)
        log = '{0}\t\t\tsensitivity: {sensitivity:.3f}\t\t\tspecificity: {specificity:.3f}'.format(raw_img_path, sensitivity=sensitivity, specificity=specificity)
    return log

def metric_iou_strategy(pil_label, pil_mask, raw_img_path):
    log = ''
    if pil_label is not None:
        iou = metric_iou(pil_label, pil_mask)
        log = '{0}\t\t\tiou: {iou:.3f}\t\t\tiou: {iou:.3f}'.format(raw_img_path, iou=iou)
    return log

metric_strategy = metric_f1_strategy if args.metrics == 'f1' else metric_iou_strategy



for raw_label in labels_256_list:
    print(raw_label)
    pil_label, pil_mask, raw_img_path = post_processing_strategy(raw_label, gt_mask_list, dict)
    log = metric_strategy(pil_label, pil_mask, raw_img_path)
    logger.append(log)
    print(log)


# if args.bboxfile is None:
#     for raw_label in labels_256_list:
#         print(raw_label)
#         # pil_label, pil_mask, raw_img_path = labels_256_postprocessing_cropped(raw_label, gt_mask_list, dict)
#         pil_label, pil_mask, raw_img_path = labels_256_postprocessing_placeholder(raw_label,gt_mask_list, dict)
#         if pil_label is not None:
#             sensitivity, specificity = metric_cal(pil_label, pil_mask)
#             log = '{0}\t\t\tsensitivity: {sensitivity:.3f}\t\t\tspecificity: {specificity:.3f}'.format(raw_img_path,
#                                                                                                        sensitivity=sensitivity,
#                                                                                                        specificity=specificity)
#             logger.append(log)
#             print(log)
# else:
#     for raw_label in labels_256_list:
#         print(raw_label)
#         # pil_label, pil_mask, raw_img_path = labels_256_postprocessing_cropped(raw_label, gt_mask_list, dict)
#         pil_label, pil_mask, raw_img_path = labels_256_postprocessing_cropped(raw_label,gt_mask_list, dict)
#         if pil_label is not None:
#             sensitivity, specificity = metric_cal(pil_label, pil_mask)
#             log = '{0}\t\t\tsensitivity: {sensitivity:.3f}\t\t\tspecificity: {specificity:.3f}'.format(raw_img_path,
#                                                                                                        sensitivity=sensitivity,
#                                                                                                        specificity=specificity)
#             logger.append(log)
#             print(log)



with open(out_log_path, 'w') as f:
    f.write('\n \n'.join(logger))