from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
import os
from gtts import gTTS

app = Flask(__name__)
model = tf.keras.models.load_model('traffic_model_vgg16.h5')

# Move class_labels to a more global scope
class_labels = {
    0: 'accident',
    1: 'dense_traffic',
    2: 'fire',
    3: 'sparse_traffic',
}

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_prediction(predicted_class):
    # Default message if the predicted_class is not recognized
    message = "Error: Unable to load and preprocess the test image."

    if predicted_class == 'accident':
        message = "Alert: Accident reported. Drive with caution."
    elif predicted_class == 'dense_traffic':
        message = "Oops!! Heavy traffic ahead. Consider an alternative route."
    elif predicted_class == 'fire':
        message = "Alert: Fire detected. Consider an alternative route!"
    elif predicted_class == 'sparse_traffic':
        message = "Yay!! No traffic. Enjoy a smooth ride."

    # Render HTML template with the prediction text
    render_result = render_template('result.html', prediction_text=message)

    # Convert the message to speech
    text_to_speech(message)

    return render_result

def text_to_speech(message):
    # Use gTTS to convert text to speech
    tts = gTTS(text=message, lang='en')
    
    # Save the speech as an audio file
    audio_file_path = 'static/audio/output.mp3'
    tts.save(audio_file_path)

    # Play the audio file (you can customize this based on your needs)
    os.system(f'start {audio_file_path}')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return render_template('result.html', prediction_text="Error: No file part")

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return render_template('result.html', prediction_text="Error: No selected file")

    # Check if the file is allowed
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        # Save the file to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Function to preprocess a test image
        def preprocess_test_image(test_image_path):
            img = cv2.imread(test_image_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img / 255.0
                return img
            else:
                return None

        # Reshape the image to match the input shape expected by the model
        test_image = preprocess_test_image(file_path)

        # Check if the image is not None
        if test_image is not None:
            # Reshape the image to match the input shape expected by the model
            test_image = np.expand_dims(test_image, axis=0)
            # Make predictions
            predictions = model.predict(test_image)

            # Map predicted class probabilities to class labels
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions)

            # Call the process_prediction function
            result = process_prediction(predicted_class)
            return result

    else:
        return render_template('result.html', prediction_text="Error: Unsupported file format")

if __name__ == '__main__':
    app.run(port=8000)
