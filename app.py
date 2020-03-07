import os
import io

import torch
import torchvision.transforms as transforms
from PIL import Image
from gevent.pywsgi import WSGIServer
from flask import Flask, request, render_template, jsonify

from models.model import Net

app = Flask('crack_detection')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'crack_detection.pth')

# Load trained model
net = Net()
net.load_state_dict(torch.load(MODEL_PATH))
print('Model loaded. Start serving...')


print('Model loaded. Check http://127.0.0.1:8080/')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_image_class():
    img = request.files['file'].read()
    img = Image.open(io.BytesIO(img)).convert('RGB')
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    tensor = transform(img).unsqueeze(0)
    outputs = net.forward(tensor)
    _, y_hat = outputs.max(1)
    prediction_classes = ('crack', 'no_crack')
    response = {"prediction": prediction_classes[y_hat]}
    return jsonify(response)


if __name__ == '__main__':
    http_server = WSGIServer(('', 8080), app)
    http_server.serve_forever()
