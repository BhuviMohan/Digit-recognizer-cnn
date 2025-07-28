import os
from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('model/mnist_cnn_model.keras')  # new fine-tuned model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Accepts a PIL image (grayscale), preprocesses it to match MNIST-style,
    and returns a 4D NumPy array ready for prediction.
    """
    img = image.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert if needed (black bg, white digit)

    img_np = np.array(img)

    # Threshold to binary image
    _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of digit
    coords = cv2.findNonZero(img_np)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop and resize to 20x20
    cropped = img_np[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)

    # Place it into center of 28x28 canvas
    final_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    final_img[y_offset:y_offset+20, x_offset:x_offset+20] = resized

    # Normalize and reshape
    final_img = final_img.astype('float32') / 255.0
    final_img = final_img.reshape(1, 28, 28, 1)

    return final_img

def predict_image(image_path):
    img = Image.open(image_path).convert('L')    # Grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)

    # Auto-detect polarity
    if np.mean(img_array) > 127:
        img = ImageOps.invert(img)
        img_array = np.array(img)

    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded'
    file = request.files['file']
    if file.filename == '':
        return 'No file selected'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predicted_class, confidence = predict_image(file_path)
        return render_template(
            'index.html',
            prediction=f"Predicted Digit: {predicted_class}",
            confidence=f"Confidence: {confidence:.2f}%",
            image_path=file_path,
            from_canvas=False
        )

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    data = request.get_json()
    img_data = data['image']
    header, imgstr = img_data.split(';base64,')
    img_bytes = base64.b64decode(imgstr)
    img = Image.open(BytesIO(img_bytes)).convert('L')

    # Save temporarily
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'canvas_digit.png')
    img.save(save_path)

    # Use improved preprocessing
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    return {
        "prediction": f"Predicted Digit: {predicted_class}",
        "confidence": f"Confidence: {confidence:.2f}%",
        "from_canvas": True
    }

if __name__ == '__main__':
    app.run(debug=True)
