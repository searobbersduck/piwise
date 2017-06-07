from glob import glob
from PIL import Image
import numpy as np

# imgs = glob('../png/*.png')
#
# im1 = np.asarray(Image.open(imgs[0]))
#
# img_big = np.zeros((256*11, 256*16, 3), dtype=np.uint8)
#
# print(img_big.dtype)
#
# for i,im in enumerate(imgs):
#     with Image.open(im) as pilim:
#         print(i)
#         im1 = np.asarray(pilim)
#         row = i // 16
#         column = i % 16
#         img_big[row*256:(row+1)*256, column*256:(column+1)*256] = im1
#
# img_big = Image.fromarray(img_big)
#
# img_big.save('./ma.png')

# im1 = Image.open('992_left.jpeg')
# im2 = Image.open('ma.png')
# im_tmp = im2.crop((0,0,im1.width, im1.height))
#
# im3 = Image.blend(im1,im_tmp,0.8)
#
# im3.save('blend.png')
#
# print(im_tmp)
