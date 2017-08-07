import http.server
import socketserver

from http.server import BaseHTTPRequestHandler
from http import HTTPStatus

import io
from io import BytesIO

from PIL import Image

PORT = 8003

class ImageHTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super(ImageHTTPRequestHandler, self).__init__(request, client_address, server)
        self.image = None

    def do_GET(self):
        print('do_GET')
        self._segment()

    def do_POST(self):
        print('do_POST')

    def _segment(self):
        data1 = self.rfile.read(int(self.headers['Content-Length']))
        stream = BytesIO(data1)
        image = Image.open(stream)
        image.show()

        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='jpeg')
        imgByteArr = imgByteArr.getvalue()

        self.send_response(HTTPStatus.OK)
        # self.send_header("Content-type", 'text/plain')
        # self.send_header("image_uid", str('111111'))
        self.send_header("Content-type", 'image/jpeg')
        self.send_header("image_uid", str('111111'))
        self.end_headers()
        self.wfile.write(imgByteArr)




Handler = ImageHTTPRequestHandler



with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        print("serving at port", PORT)
        httpd.serve_forever()
    finally:
        print('finish')