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

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

app = Flask(__name__)
CORS(app)


def get_model():
    global model
    K.clear_session()
    model = VGG16(weights='imagenet', include_top=True)

@app.route('/predict', methods=['POST'])
def predict():
    get_model()
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
    label = decode_predictions(prediction, top=10)[0]
    label = ({b: c} for a, b, c in label)
    values = ''.join(str(v) for v in label)

    return jsonify(values)
