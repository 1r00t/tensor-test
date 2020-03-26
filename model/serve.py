from flask import Flask, jsonify
from flask_restful import Resource, Api
from tensorflow import keras
from matplotlib import image
import numpy as np


app = Flask(__name__)
api = Api(app)

def load_model():
    model = keras.models.model_from_json(open("mymodel.json", "r").read())
    model.load_weights("myweights.h5")
    probability_model = keras.Sequential([
        model,
        keras.layers.Softmax()
    ])
    return probability_model

def img_to_gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.144])

def predict(model, img):
    img = img_to_gray(img)
    img = img / 255.0
    return model.predict([[img]])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = load_model()

@app.route("/", methods=["GET"])
def index():
    img = image.imread("../images/sneakernike-28.png")
    prediction = predict(model, img)
    prediction = np.argmax(prediction[0])
    return jsonify({"prediction": class_names[prediction]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
