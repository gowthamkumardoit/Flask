import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169

from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.applications.densenet import preprocess_input, decode_predictions

app = Flask(__name__)
CORS(app)

# VGG 16 Model
def get_model_vgg16():
    global model
    K.clear_session()
    model = VGG16(weights='imagenet', include_top=True)

@app.route('/predict', methods=['POST'])
def predictVGG16():
    get_model_vgg16()
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = 224
    height = 224
    image = image.resize((width, height), Image.NEAREST)
    image = np.expand_dims(image, axis=0)
    x = preprocess_input(image)

    prediction = model.predict(x)
    label = decode_predictions(prediction, top=5)[0]
    label = ({b: c} for a, b, c in label)
    values = ''.join(str(v) for v in label)

    return jsonify(values)



# VGG 19 Model
def get_model_vgg19():
    global model
    K.clear_session()
    model = VGG16(weights='imagenet', include_top=True)

@app.route('/predictvgg', methods=['POST'])
def predictVGG19():
    get_model_vgg19()
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = 224
    height = 224
    image = image.resize((width, height), Image.NEAREST)
    image = np.expand_dims(image, axis=0)
    x = preprocess_input(image)

    prediction = model.predict(x)
    label = decode_predictions(prediction, top=5)[0]
    label = ({b: c} for a, b, c in label)
    values = ''.join(str(v) for v in label)

    return jsonify(values)




# ResNet Model
def get_model_resNet():
    global model
    K.clear_session()
    model = ResNet50(weights='imagenet', include_top=True)

@app.route('/predictResNet', methods=['POST'])
def predictResNet():
    get_model_resNet()
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = 224
    height = 224
    image = image.resize((width, height), Image.NEAREST)
    image = np.expand_dims(image, axis=0)
    x = preprocess_input(image)

    prediction = model.predict(x)
    label = decode_predictions(prediction, top=5)[0]
    label = ({b: c} for a, b, c in label)
    values = ''.join(str(v) for v in label)

    return jsonify(values)

# Inception Model
def get_model_inception():
    global model
    K.clear_session()
    model = InceptionV3(weights='imagenet', include_top=True)

@app.route('/predictInception', methods=['POST'])
def predictInception():
    get_model_inception()
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = 299
    height = 299
    image = image.resize((width, height), Image.NEAREST)
    image = np.expand_dims(image, axis=0)
    x = preprocess_input(image)

    prediction = model.predict(x)
    label = decode_predictions(prediction, top=5)[0]
    label = ({b: c} for a, b, c in label)
    values = ''.join(str(v) for v in label)

    return jsonify(values)

# DenseNet121 Model
def get_model_denseNet121():
    global model
    K.clear_session()
    model = DenseNet121(weights='imagenet', include_top=True)

@app.route('/predictDenseNet121', methods=['POST'])
def predictDenseNet121():
    get_model_denseNet121()
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = 299
    height = 299
    image = image.resize((width, height), Image.NEAREST)
    image = np.expand_dims(image, axis=0)
    x = preprocess_input(image)

    prediction = model.predict(x)
    label = decode_predictions(prediction, top=5)[0]
    label = ({b: c} for a, b, c in label)
    values = ''.join(str(v) for v in label)

    return jsonify(values)


# DenseNet169 Model
def get_model_denseNet169():
    global model
    K.clear_session()
    model = DenseNet169(weights='imagenet', include_top=True)

@app.route('/predictDenseNet169', methods=['POST'])
def predictDenseNet169():
    get_model_denseNet169()
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = 299
    height = 299
    image = image.resize((width, height), Image.NEAREST)
    image = np.expand_dims(image, axis=0)
    x = preprocess_input(image)

    prediction = model.predict(x)
    label = decode_predictions(prediction, top=5)[0]
    label = ({b: c} for a, b, c in label)
    values = ''.join(str(v) for v in label)

    return jsonify(values)