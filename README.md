**Digit Recognizer Web App**
A simple web application that recognizes handwritten digits using a trained machine learning model.

**Features**
Draw digits on a canvas and let the model predict the number.
User-friendly web interface for live digit recognition.
Fast and accurate predictions powered by a convolutional neural network.
Easily extensible for further improvements or educational use.

Getting Started
Prerequisites
Python 3.x

All required libraries listed in requirements.txt

**Installation**
1. Clone the Repository
git clone https://github.com/BhuviMohan/Digit-recognizer-cnn.git
cd Digit-recognizer-cnn

2. Backend Setup
Install Python dependencies:
pip install -r requirements.txt

3. Frontend Setup (if applicable)
cd frontend
npm install
npm run build
Running the App

To start the application:
python app.py
Then open your browser and visit: http://localhost:5000

**Usage**
Draw a digit (0–9) on the canvas.

Click Predict to view the model's prediction.

**Project Structure**
.
├── app.py               # Main backend server
├── model/               # Trained models and related files
├── static/              # Static assets (CSS, JS, Images)
├── templates/           # HTML templates
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
**Model**
The model is trained on the MNIST dataset to recognize handwritten digits (0–9) using a Convolutional Neural Network (CNN).

**Contributing**
Contributions are welcome. If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

**License**
This project is licensed under the MIT License. See the LICENSE file for more information.

**Acknowledgments**
MNIST Dataset
Libraries and frameworks used: TensorFlow/Keras, Flask, NumPy, OpenCV, and others.
