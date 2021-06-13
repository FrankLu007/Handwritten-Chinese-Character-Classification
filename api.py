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
# torch.multiprocessing.set_start_method('spawn', force = True)
from torch.multiprocessing import Process, Queue
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
        image.save('public4/' + name + '_' + prediction + '.jpg')

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force = True)
    print('get', time() - data['esun_timestamp'])

    with torch.no_grad():
        torch.cuda.empty_cache()
        image = ToTensor(Image.open(BytesIO(b64decode(data['image'])))).half().cuda().unsqueeze_(0)
        inputs = test_transform(image)
        model_id = model_usage.get(True)
        outputs = model[model_id](inputs)[0]
        model_usage.put(model_id, False)
        prediction = classes[outputs.argmax(0)]
        del inputs, outputs, image
        
    image_storage.put((data['esun_uuid'], data['image'], prediction), False)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    s = sha256()
    s.update((CAPTAIN_EMAIL + ts + SALT).encode("utf-8"))
    server_uuid = s.hexdigest()
    print('end', prediction, time() - data['esun_timestamp'])
    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': prediction,
                    'server_timestamp': time()})


if __name__ == "__main__":

    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=5000, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    options = arg_parser.parse_args()

    image_storage = Queue()
    model_usage = Queue()
    model_usage.put(0)
    model_usage.put(1)
    model_usage.put(2)
    model_usage.put(3)
    
    with open('class.txt', 'r', encoding = 'utf8') as file:
        classes = file.read().splitlines()
    classes.append('isnull')


    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        test = torch.zeros((1, 3, 448, 448)).half().cuda()
        model = [torch.load('eff_b8_448.weight') for _ in range(4)]
        # model = [torch.jit.trace(torch.load('eff_b8_448.weight'), test, strict = False) for _ in range(4)]
        del test
    for m in model:
        m.eval()
    # model.eval()
    test_transform = torch.nn.Sequential(
        transforms.Pad((60, 30), fill = 0.941176),
        transforms.CenterCrop((67, 100)),
        transforms.Resize((448, 448)),
        transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
    )
    addition_transform = torch.nn.Sequential(
        transforms.Pad((5, 5), fill = 0.941176),
        transforms.RandomRotation(10, fill = 0.941176),
        transforms.RandomPerspective(0.1, fill = 0.941176),
        transforms.RandomCrop((67, 100), fill = 0.941176, pad_if_needed = True),
        transforms.Resize((448, 448)),
        transforms.ColorJitter(brightness = (0.8, 1.2), saturation = (0.8, 1.2), contrast = (0.8, 1.2), hue = 0.05),
        transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
    )

    ToTensor = transforms.ToTensor()
    p = Process(target = save_image, args = (image_storage, ), daemon = True)
    p.start()
    
    app.run(host='0.0.0.0', port = options.port, debug = options.debug, threaded = True)

