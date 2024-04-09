from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('potato_disease_model.h5')

# Define the classes
classes = {0: 'Early Blight', 1: 'Healthy', 2: 'Light Blight'}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the request
        img_file = request.files['file']
        
        # Read the image file
        img = image.load_img(io.BytesIO(img_file.read()), target_size=(256, 256))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        
        # Expand the dimensions of the image array
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make the prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Get the class label
        predicted_label = classes[predicted_class[0]]
        
        # Redirect to the predict page with the predicted label and image file
        return redirect(url_for('predict', label=predicted_label, image_file=img_file))
    else:
        return render_template('index.html')

@app.route('/predict')
def predict():
    # Get the predicted label and image file from the query parameters
    predicted_label = request.args.get('label')
    img_file = request.args.get('image_file')

     # Check the predicted label and set the message accordingly
    if predicted_label == 'Healthy':
        message = 'The potato leaf plant uploaded is healthy and does not contain any diseases.'
    else:
        message = f'The potato leaf plant uploaded contains {predicted_label} disease.'
    # Render the predict page with the predicted label and image file
    return render_template('predict.html', message = message, image_path=img_file)
