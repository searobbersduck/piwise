file = '/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata_patches/labels/C0000887_3_4.png'

from PIL import Image
import numpy as np
import torch

img1 = Image.open(file)
img1_p = Image.open(file).convert('P')
img1_l = Image.open(file).convert('L')

img11 = np.array(img1)
img12 = torch.from_numpy(img11)
img13 = img12.long()
img14 = img13.unsqueeze(0)

print('hi')

import visdom

vis = visdom.Visdom()

val = np.array(img1)

val = val[np.newaxis, :]

val1 = np.random.randint(0,255,size=(3,256,256), dtype=np.uint8)

vis.image(val1)


import numpy as np

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

n = 22

cmap = colormap(256)

cmap[n] = cmap[-1]

from torch.autograd import Variable
import torch

a = 3

a1 = torch.Tensor(a)

a2 = Variable(a1)

b1 = torch.rand((4,4,5,5))
b2 = Variable(b1)
b3 = b2[0]
b4 = b3.max(0)
b5 = b4[1].data

mask = b5[0] == 2

print(cmap)