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
from keras.models import model_from_json

import pandas as pd
from classify import c100_classify
app = Flask(__name__)
CORS(app)


def get_model():
    global model
    K.clear_session()
    model = load_model('./models/cifar_10_final.h5')
    model._make_predict_function()

get_model()
@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    width = 32
    height = 32
    image = image.resize((width, height), Image.NEAREST)
    im2arr = np.array(image)/255.  # im2arr.shape: height x width x channel

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
   
    image = np.expand_dims(im2arr, axis=0)
    prob = model.predict_proba(np.reshape(image,(1,32,32,3)), batch_size=1, verbose=0)
    pred = pd.DataFrame(data = np.reshape(prob,10), index=labels, columns={'probability'}).sort_values('probability', ascending=False)
    pred['name'] = pred.index
    data = [{'name':x, 'probability':y} for x,y in zip(pred.iloc[:5,1],pred.iloc[:5,0])]
    print(data)
    return jsonify(data)
