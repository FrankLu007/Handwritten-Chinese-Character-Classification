from argparse import ArgumentParser
from PIL import Image
from io import BytesIO
from base64 import b64decode
from hashlib import sha256
from time import time
from flask import Flask, request, jsonify, _request_ctx_stack
import torchvision.transforms as transforms
import datetime
import torch, timm
from torch.multiprocessing import Process, Queue

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'shes990106@gmail.com' #
SALT = 'IamChengHanLuFromNTUCSAI'       #
#########################################
def save_image(que):

    while True:
        name, image_b64, prediction = que.get(True)
        image = Image.open(BytesIO(b64decode(image_b64)))
        image.save('private4/' + name + '_' + prediction + '.jpg')

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force = True)

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
    arg_parser.add_argument('-sz', '--size', default=448)
    arg_parser.add_argument('-n', '--n_thread', default=4)
    options = arg_parser.parse_args()

    image_storage = Queue()
    model_usage = Queue()
    for i in range(options.n_thread):
        model_usage.put(i)
    
    with open('class.txt', 'r', encoding = 'utf8') as file:
        classes = file.read().splitlines()
    classes.append('isnull')


    torch.backends.cudnn.benchmark = True
    model = [torch.load('tmp.weight').half() for _ in range(options.n_thread)]
    for m in model:
        m.eval()

    test_transform = torch.nn.Sequential(
        transforms.Pad((60, 30), fill = 0.941176),
        transforms.CenterCrop((67, 100)),
        transforms.Resize((448, 448)),
        transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
    )

    ToTensor = transforms.ToTensor()
    p = Process(target = save_image, args = (image_storage, ), daemon = True)
    p.start()
    
    app.run(host='0.0.0.0', port = options.port, debug = options.debug, threaded = True)

