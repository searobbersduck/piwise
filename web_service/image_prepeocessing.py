from PIL import Image

from skimage.filters import threshold_otsu
from skimage import measure, exposure

import skimage

import scipy.misc

import numpy as np

import torch

import sys

sys.path.append('..')

from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet
from torchvision.transforms import Compose, CenterCrop, Scale, Normalize, ToTensor

__all__ = [

]



NUM_CHANNELS = 3
NUM_CLASSES = 2

eval_input_transform = Compose([
    Scale(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])

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


def main():
    pil_img = Image.open('/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata/images/C0000887.jpg')
    # pil_patch = get_patch(pil_img, 1000,1000, 256, 256)
    # pil_patch = get_ahe_patch(pil_img, 1000,1000, 256, 256)
    # p1 = pil_patch
    # pil_patch.show()
    # p1 = p1.resize((128,128))
    # pil_patch.show()
    # p1.show()
    seg = DRImageSegmentor('fcn8', '1.pth', eval_input_transform)
    seg.segment(pil_img, 1000,1000, 256, 256)


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
        np_label = o_label.numpy()
        pil_label = Image.fromarray(np_label)
        # pil_label.show()
        pil_label.save('test.png')



if __name__ == '__main__':
    main()