import base64

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template, jsonify,send_file
import os
import io
app = Flask(__name__)


model_path = '/Users/amananand/PycharmProjects/fire-detection/final_classification_model/final_model.h5'
model = tf.keras.models.load_model(model_path)


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict_image(img_array):
    predictions = model.predict(img_array)
    if predictions[0][0] > predictions[0][1]:
        return "Fire", predictions[0][0], predictions[0][1]
    else:
        return "Non-Fire", predictions[0][0], predictions[0][1]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Read the image from the request
        img_data = request.data
        img = Image.open(io.BytesIO(img_data))

        # Preprocess the image
        img_array = preprocess_image(img)

        # Make a prediction
        prediction, confidence_fire, confidence_non_fire = predict_image(img_array)

        confidence_fire = float(confidence_fire)
        confidence_non_fire = float(confidence_non_fire)

        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # Return the prediction as a JSON response
        return jsonify({
            'prediction': prediction,
            'confidence_fire': confidence_fire,
            'confidence_non_fire': confidence_non_fire,
            'image': img_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
        app.run(debug=True)