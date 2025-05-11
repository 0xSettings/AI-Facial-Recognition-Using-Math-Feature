import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def extract_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        return gray[y:y+h, x:x+w]
    return None

def prepare_dataset(dataset_path):
    X, y = [], []
    label_dict = {}
    label = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        label_dict[label] = person_name
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            face = extract_face(image_path)
            if face is not None:
                face_resized = cv2.resize(face, (100, 100)).flatten()
                X.append(face_resized)
                y.append(label)
        label += 1

    return np.array(X), np.array(y), label_dict

def train_models(X, y):
    if len(X) == 0:
        raise ValueError("Training data is empty. Check your dataset.")
    
    # PCA components 
    n_components = min(50, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # LDA components n_classes - 1 components
    lda_components = min(len(np.unique(y)) - 1, X_pca.shape[1])
    lda = LDA(n_components=lda_components)
    X_lda = lda.fit_transform(X_pca, y)

    return pca, lda, X_lda


def recognize_face(image_path, pca, lda, X_lda, y, label_dict):
    face = extract_face(image_path)
    if face is None:
        return "No face detected"
    face_resized = cv2.resize(face, (100, 100)).flatten()
    face_pca = pca.transform([face_resized])
    face_lda = lda.transform(face_pca)

    distances = np.linalg.norm(X_lda - face_lda, axis=1)
    min_index = np.argmin(distances)
    predicted_label = y[min_index]
    return label_dict[predicted_label]
