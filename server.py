from http.server import HTTPServer, BaseHTTPRequestHandler

import sys, os

cwd = os.getcwd()
sys.path.append(cwd)

from io import BytesIO
import json

import vqa_predict

class HTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        # read json
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)
        image = data['image']
        question = data['question']        
        answer = vqa_predict.predict_preprocess(image, question)
      
        # write json
        data['answer'] = answer
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(bytes(json.dumps(data),'utf-8'))
        print(question)
        print(answer)

    def do_GET(self):
        do_POST(self)

if __name__ == '__main__':
    print("starting server")
    httpd = HTTPServer(('192.168.1.60', 8080), HTTPRequestHandler)
    httpd.serve_forever()

