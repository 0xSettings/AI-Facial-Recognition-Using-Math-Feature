from flask import Flask, render_template, request
from preprocess import preprocess_image
from extract_math import apply_pca, apply_lda
from svm_classifier import predict_svm
import numpy as np
import os
import pickle

app = Flask(__name__)

with open('pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)

with open('lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)
            features = preprocess_image(filepath)
            test_pca = pca_model.transform([features])
            test_lda = lda_model.transform(test_pca)
            pred, conf = predict_svm(svm_model, test_lda)
            return f"Prediction: {pred[0]} - Confidence: {conf[0][pred[0]]:.2f}"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)