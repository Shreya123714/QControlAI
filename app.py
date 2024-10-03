from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the fine-tuned model
model = load_model('path_to_your_model.h5')

# Define a function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match model input shape
    image = image / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        if request.files:
            image_file = request.files['image']
            # Save the uploaded image temporarily
            image_path = os.path.join("uploads", image_file.filename)
            image_file.save(image_path)

            # Load and preprocess the image
            image = cv2.imread(image_path)
            preprocessed_image = preprocess_image(image)

            # Make a prediction
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions, axis=1)

            # Interpret the result (you need to map your class index to class labels)
            if predicted_class[0] == 0:
                prediction = "No defect detected."
            else:
                prediction = "Defect detected!"

            # Optionally, remove the uploaded image after prediction
            os.remove(image_path)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
