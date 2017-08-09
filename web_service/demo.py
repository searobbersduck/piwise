import http.client

from PIL import Image

from io import BytesIO


conn = http.client.HTTPConnection("face.baidu.com")


patch = '/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata/images/C0001273.jpg'

data = open(patch, 'rb').read()

img_ref = Image.open(patch)

print('raw image patch size is: {}'.format(img_ref.size))

headers = {"Content-type": "image/jpeg", "Accept": "q=0.6, image/jpeg", "Content-Length": str(len(data)), "coord":"1000 1000 256 256",}


conn.request('GET', "/test/for/med/segment", data, headers)


r = conn.getresponse()

image_uid = r.headers['image_uid']

print('image uid: {}'.format(image_uid))

img = r.read()

img = BytesIO(img)

pil_img = Image.open(img)

print('image label size is: {}'.format(pil_img.size))

pil_img.show()



print('end')

conn.close()