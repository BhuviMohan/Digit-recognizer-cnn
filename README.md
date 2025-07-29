Digit Recognizer Web App
A simple web application that recognizes handwritten digits using a trained machine learning model.

Features
Draw digits on a canvas and let the model predict the number.
User-friendly web interface for live digit recognition.
Fast and accurate predictions powered by machine learning.
Easily extensible for further improvements or educational use.

Getting Started
Prerequisites
Python 3.x
Required libraries (see Installation)

Installation
Clone the repository

bash
git clone https://github.com/BhuviMohan/Digit-recognizer-cnn.git
cd digit-recognizer-cnn
(Backend) Install Python dependencies

bash
pip install -r requirements.txt
(Frontend) Install frontend dependencies (if any)

bash
cd frontend
npm install
npm run build
Run the app

bash
python app.py
Or, according to your framework’s requirements.

Open your browser and visit: http://localhost:5000

Usage
Draw a digit (0-9) on the canvas.

Click "Predict" to see the model's prediction.

(Optional) Save screenshots or model outputs.

Project Structure
text
.
├── app.py               # Main backend server
├── model/               # Trained models and related files
├── static/              # Static assets (HTML, CSS, JS)
├── templates/           # HTML templates
├── requirements.txt     # Python dependencies
└── README.md            # This file
Model
Trained on the MNIST dataset or similar for handwritten digit recognition.

Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
MNIST Dataset

Any libraries or frameworks used
