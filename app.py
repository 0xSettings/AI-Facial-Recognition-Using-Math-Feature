from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from face_recognition_module import prepare_dataset, train_models, recognize_face
app = Flask(__name__, template_folder='frontend')


app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load and train models
X, y, label_dict = prepare_dataset('dataset')
pca, lda, X_lda = train_models(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = recognize_face(filepath, pca, lda, X_lda, y, label_dict)
    return render_template('index.html', result=result, image_url='/' + filepath)

if __name__ == '__main__':
    app.run(debug=True)
