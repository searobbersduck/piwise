import http.client

from PIL import Image

from io import BytesIO

conn = http.client.HTTPConnection('127.0.0.1', port=8003)

data = open('/Users/zhangweidong03/Code/dl/pytorch/github/pi/piwise/MAdata/images/C0000886.jpg', 'rb').read()

headers = {"Content-type": "image/jpeg", "Accept": "q=0.6, image/jpeg", "Content-Length": str(len(data))}

conn.request('GET', "", data, headers)

r = conn.getresponse()

image_uid = r.headers['image_uid']

print('image uid: {}'.format(image_uid))

img = r.read()

img = BytesIO(img)

pil_img = Image.open(img)

pil_img.show()

# headers = {"Content-type": "text/plain", "coord":"1000 1000 256 256", "image_uid":image_uid}
#
# conn.request('GET', "", "", headers=headers)
#
# r = conn.getresponse()

print('end')

conn.close()