import http.client

from PIL import Image

from io import BytesIO

# conn = http.client.HTTPConnection('127.0.0.1', port=8003)

conn = http.client.HTTPConnection('yq01-idl-gpu-online9.yq01.baidu.com', port=8002)


# data = open('/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata/images/C0000886.jpg', 'rb').read()

patch = '/Users/zhangweidong03/data/ex/ex_patches/images/C0024407_EX_4_3.png'

data = open(patch, 'rb').read()

img_ref = Image.open(patch)
print('raw image patch size is: {}'.format(img_ref.size))

headers = {"Content-type": "image/jpeg", "Accept": "q=0.6, image/jpeg", "Content-Length": str(len(data))}

conn.request('GET', "", data, headers)

r = conn.getresponse()

image_uid = r.headers['image_uid']

print('image uid: {}'.format(image_uid))

img = r.read()

img = BytesIO(img)

pil_img = Image.open(img)

print('image label size is: {}'.format(pil_img.size))

pil_img.show()

# headers = {"Content-type": "text/plain", "coord":"1000 1000 256 256", "image_uid":image_uid}
#
# conn.request('GET', "", "", headers=headers)
#
# r = conn.getresponse()

print('end')

conn.close()