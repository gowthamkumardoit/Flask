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


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import cv2 as cv

app = Flask(__name__)
CORS(app)


def get_model():
    global model
    K.clear_session()
    model = load_model('cifar10.h5')


@app.route('/predict', methods=['POST'])
def predict():
    get_model()
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = 32
    height = 32
    image = image.resize((width, height), Image.NEAREST)
    im2arr = np.array(image)  # im2arr.shape: height x width x channel
    print(im2arr)
    arr2im = im2arr.reshape(3, 32, 32)
    print(arr2im.shape)
    #arr2im = Image.fromarray(im2arr)
    # print(im2arr)
    #im = cv.resize(image,  (width, height))
    #im.reshape((height, height))
    # print(im.shape)  # (28,28)

    #image = cv2.resize((width, height), Image.NEAREST)
    image = np.expand_dims(arr2im, axis=0)

    print(image.shape)
    #x = preprocess_input(image)

    prediction = model.predict(image)[0]
    print(prediction)
    values = ','.join(str(v) for v in prediction)
    # label = decode_predictions(prediction, top=10)[0]
    # label = ({b: c} for a, b, c in label)
    # values = ''.join(str(v) for v in label)

    return jsonify(values)
