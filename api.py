from argparse import ArgumentParser
from PIL import Image
from io import BytesIO
from base64 import b64decode
from hashlib import sha256
from time import time
from flask import Flask, request, jsonify, _request_ctx_stack
import torchvision.transforms as transforms
import datetime
import torch
from torch.multiprocessing import Process, Queue
from transformers import ViTForImageClassification
from os import getpid

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'shes990106@gmail.com' #
SALT = 'IamChengHanLuFromNTUCSAI'       #
#########################################
def save_image(que):

    while True:
        name, image_b64, prediction = que.get(True)
        image = Image.open(BytesIO(b64decode(image_b64)))
        image.save('public3/' + name + '_' + prediction + '.jpg')

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force = True)
    print('get', time() - data['esun_timestamp'])
    print(_request_ctx_stack._local.__ident_func__())
    print(getpid())
    while True:
        s = 1
    # with torch.no_grad():
    #     torch.cuda.empty_cache()
    #     image = ToTensor(Image.open(BytesIO(b64decode(data['image'])))).half().cuda().unsqueeze_(0).repeat(64, 1, 1, 1)
    #     inputs = addition_transform(image)
    #     outputs = model(inputs)['logits'].mean(0)
    #     prediction = classes[outputs.argmax(0)]
    #     del inputs, outputs, image
    # image_storage.put((data['esun_uuid'], data['image'], prediction), False)

    # t = datetime.datetime.now()
    # ts = str(int(t.utcnow().timestamp()))
    # s = sha256()
    # s.update((CAPTAIN_EMAIL + ts + SALT).encode("utf-8"))
    # server_uuid = s.hexdigest()
    # print('end', prediction, time() - data['esun_timestamp'])
    # return jsonify({'esun_uuid': data['esun_uuid'],
    #                 'server_uuid': server_uuid,
    #                 'answer': prediction,
    #                 'server_timestamp': time()})


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=5000, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    options = arg_parser.parse_args()

    image_storage = Queue()
    mp.set_start_method('spawn')
    # with open('class.txt', 'r', encoding = 'utf8') as file:
    #     classes = file.read().splitlines()
    # classes.append('isnull')


    # torch.backends.cudnn.benchmark = True
    # with torch.no_grad():
    #     test = torch.zeros((64, 3, 384, 384)).half().cuda()
    #     model = torch.jit.trace(torch.load('tmp.weight'), test, strict = False)
    #     del test
    # model.eval()
    
    # addition_transform = torch.jit.script(torch.nn.Sequential(
    #     transforms.Pad((60, 0), padding_mode = 'edge'),
    #     transforms.RandomRotation(10, fill = 0.9411),
    #     transforms.CenterCrop((67, 100)),
    #     transforms.Resize((384, 384)),
    #     transforms.GaussianBlur(3, 2.0),
    #     transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
    #     transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
    # ))

    # ToTensor = transforms.ToTensor()
    p = Process(target = save_image, args = (image_storage, ), daemon = True)
    p.start()
    
    app.run(host='0.0.0.0', port = options.port, debug = options.debug, threaded = False, processes = 4)

