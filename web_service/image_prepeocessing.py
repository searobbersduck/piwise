from PIL import Image

from skimage.filters import threshold_otsu
from skimage import measure, exposure

import skimage

import scipy.misc

import numpy as np

import torch

import cv2

import sys

sys.path.append('..')

from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet
from torchvision.transforms import Compose, CenterCrop, Scale, Normalize, ToTensor, ToPILImage

__all__ = [
    'DRImageSegmentor',
    'eval_input_transform',
    'get_segmentor',
]




NUM_CHANNELS = 3
NUM_CLASSES = 2

eval_input_transform = Compose([
    Scale(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])

image_transform = ToPILImage()

def tight_crop(img, size=None):
    img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:])+1

    img_circ = (img_label == largest_label)
    img_xs = np.sum(img_circ, 0)
    img_ys = np.sum(img_circ, 1)
    xs = np.where(img_xs>0)
    ys = np.where(img_ys>0)
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    img_crop = img[y_lo:y_hi, x_lo:x_hi, :]

    return img_crop


# adaptive historgram equlization
def channelwise_ahe(img):
    img_ahe = img.copy()
    for i in range(img.shape[2]):
        img_ahe[:,:,i] = exposure.equalize_adapthist(img[:,:,i], clip_limit=0.03)
    return img_ahe

# raw is PIL Image
def get_ahe_patch(raw, x, y, w, h):
    patch = raw.crop((x,y,x+w,y+h))
    patch = np.array(patch, dtype=np.float32)
    patch /= 255
    ahe_patch = channelwise_ahe(patch)
    pil_patch = Image.fromarray(skimage.util.img_as_ubyte(ahe_patch))
    return pil_patch

def get_patch(raw, x, y, w, h):
    patch = raw.crop((x, y, x + w, y + h))
    return patch

def get_ahe_patch(raw):
    patch = raw
    patch = np.array(patch, dtype=np.float32)
    patch /= 255
    ahe_patch = channelwise_ahe(patch)
    pil_patch = Image.fromarray(skimage.util.img_as_ubyte(ahe_patch))
    return pil_patch

def erode_and_dilate(pil_img):
    cv_label = np.array(pil_img)
    thresh, cv_label = cv2.threshold(cv_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((4, 4), np.uint8)  # 生成一个6x6的核
    erosion = cv2.erode(cv_label, kernel, iterations=1)  # 调用腐蚀算法
    dilation = cv2.dilate(erosion, kernel, iterations=1)  # 调用膨胀算法
    pil_label = Image.fromarray(dilation)
    return pil_label


def main():
    pil_img = Image.open('1.png')
    seg = DRImageSegmentor('fcn8', '1.pth', eval_input_transform)
    seg.segment(pil_img, 0,0, 256, 256)



# this class use cuda as default
class DRImageSegmentor(object):
    def __init__(self, arch, weights, eval_input_transform, devs=[0]):
        self.arch = arch
        self.weights = weights
        self.devs = devs
        self.model_loaded = False
        self.trans = eval_input_transform
        print('DRImageSegmentor initialized!!!')

    def load_model(self, weights):
        Net = FCN8
        model = Net(NUM_CLASSES)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(weights))
        return model

    def segment(self, image, x, y, w, h):
        if not self.model_loaded:
            self.model = self.load_model(self.weights)
            self.model.eval()
            self.model_loaded = True
        raw_patch = get_ahe_patch(image, x, y, w, h)
        # resize_patch = raw_patch.resize((256,256), resample=Image.BILINEAR)
        input = torch.stack([self.trans(raw_patch)])
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        output = self.model(input_var)
        o_label = output[0].cpu().max(0)[1].data
        o_label_b = o_label.type(torch.ByteTensor)
        o_label_b *= 255
        pil_label = image_transform(o_label_b)
        pil_label.resize(raw_patch.size)
        # pil_label = Image.fromarray(np_label)
        # pil_label.show()
        # pil_label.save('test.png')
        return pil_label

    def segment(self, image):
        if not self.model_loaded:
            self.model = self.load_model(self.weights)
            self.model.eval()
            self.model_loaded = True
        raw_patch = get_ahe_patch(image)
        print('raw ahe image size is: {}'.format(raw_patch.size))
        # resize_patch = raw_patch.resize((256,256), resample=Image.BILINEAR)
        input = torch.stack([self.trans(raw_patch)])
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        output = self.model(input_var)
        o_label = output[0].cpu().max(0)[1].data
        o_label_b = o_label.type(torch.ByteTensor)
        o_label_b *= 255
        pil_label = image_transform(o_label_b)
        print('image label resized to: {}'.format(raw_patch.size))
        pil_label = pil_label.resize(raw_patch.size, resample=Image.BICUBIC)
        print('image label resized to: {}'.format(pil_label.size))
        # pil_label = Image.fromarray(np_label)
        # pil_label.show()
        # pil_label.save('test.png')
        pil_label = erode_and_dilate(pil_label)
        pil_label = raw_patch
        return pil_label


def get_segmentor():
    return DRImageSegmentor('fcn8', '1.pth', eval_input_transform)


if __name__ == '__main__':
    main()