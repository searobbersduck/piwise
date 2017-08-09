import http.server
import socketserver

from http.server import BaseHTTPRequestHandler
from http import HTTPStatus


from PIL import Image

from io import BytesIO

import io

import imagehash

PORT = 8002

import sys

sys.path.append('..')

from web_service.image_prepeocessing import get_segmentor

seg = get_segmentor()

class ImageHTTPRequestHandler(BaseHTTPRequestHandler):

    """Simple HTTP request handler with GET and HEAD commands.

    This serves files from the current directory and any of its
    subdirectories.  The MIME type for files is determined by
    calling the .guess_type() method.

    The GET and HEAD requests are identical except that the HEAD
    request omits the actual contents of the file.

    """

    def __init__(self, request, client_address, server):
        print('111111111')
        self.image = 0
        self.image_id = 0
        super(ImageHTTPRequestHandler, self).__init__(request, client_address, server)


    def do_GET(self):
        print('Content type: {0}'.format(self.headers['Content-type']))
        # if self.headers['Content-type'] == 'image/jpeg':
        #     self._record()
        # elif self.headers['Content-type'] == 'text/plain':
        #     print('begin outside _segment')
        #     self._segment()
        # self._segment()
        self._segment_region()


    def _segment(self):
        data1 = self.rfile.read(int(self.headers['Content-Length']))
        stream = BytesIO(data1)
        image = Image.open(stream)
        # coord = self.headers['coord']
        # print('coord: '+ coord)
        # coords = coord.split(' ')
        # x = int(coords[1])
        # y = int(coords[2])
        # w = int(coords[3])
        # h = int(coords[4])
        # print('x:{0}/y:{1}/w:{2}/h:{3}'.format(x, y, w, h))
        pil_image = seg.segment(image)
        imgByteArr = io.BytesIO()
        pil_image.save(imgByteArr, format='jpeg')
        imgByteArr = imgByteArr.getvalue()
        self.send_response(HTTPStatus.OK)
        # self.send_header("Content-type", 'text/plain')
        # self.send_header("image_uid", str('111111'))
        self.send_header("Content-type", 'image/jpeg')
        self.send_header("image_uid", str('111111'))
        self.end_headers()
        self.wfile.write(imgByteArr)

    def _segment_region(self):
        data1 = self.rfile.read(int(self.headers['Content-Length']))
        stream = BytesIO(data1)
        image = Image.open(stream)
        coord = self.headers['coord']
        print('coord: '+ coord)
        coords = coord.split(' ')
        x = int(coords[0])
        y = int(coords[1])
        w = int(coords[2])
        h = int(coords[3])
        print('x:{0}/y:{1}/w:{2}/h:{3}'.format(x, y, w, h))
        pil_image = seg.segment(image,x,y,w,h)
        imgByteArr = io.BytesIO()
        pil_image.save(imgByteArr, format='jpeg')
        imgByteArr = imgByteArr.getvalue()
        self.send_response(HTTPStatus.OK)
        # self.send_header("Content-type", 'text/plain')
        # self.send_header("image_uid", str('111111'))
        self.send_header("Content-type", 'image/jpeg')
        self.send_header("image_uid", str('111111'))
        self.end_headers()
        self.wfile.write(imgByteArr)





        # def _record(self):
    #     data1 = self.rfile.read(int(self.headers['Content-Length']))
    #     stream = BytesIO(data1)
    #     self.image = Image.open(stream)
    #     image_id = imagehash.average_hash(self.image)
    #     self.image_id = image_id
    #     print('record image: {}'.format(self.image_id))
    #     self.send_response(HTTPStatus.OK)
    #     self.send_header("Content-type", 'text/plain')
    #     self.send_header("image_uid", str(image_id))
    #     self.end_headers()
    #
    #
    # def _segment(self):
    #     data1 = self.rfile.read(int(self.headers['Content-Length']))
    #     stream = BytesIO(data1)
    #     self.image = Image.open(stream)
    #     image_uid = self.headers['image_uid']
    #     if image_uid != self.image_id:
    #         print('image is not right!!!')
    #         self.send_response(HTTPStatus.NO_CONTENT)
    #     else:
    #         coord = self.headers['coord']
    #         print('coord: '+ coord)
    #         coords = coord.split(' ')
    #         x = int(coords[1])
    #         y = int(coords[2])
    #         w = int(coords[3])
    #         h = int(coords[4])
    #         print('x:{0}/y:{1}/w:{2}/h:{3}'.format(x, y, w, h))







Handler = ImageHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        print("serving at port", PORT)
        httpd.serve_forever()
    finally:
        print('finish')