from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load your ML model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(data)])
    return render_template('index.html', prediction_text=f'Recommended Crop: {prediction[0]}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # required for Render
    app.run(host='0.0.0.0', port=port)

