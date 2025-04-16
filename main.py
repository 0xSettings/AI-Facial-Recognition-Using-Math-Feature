from preprocess import preprocess_image
from extract_math import apply_pca, apply_lda
from svm_classifier import train_svm, predict_svm
import os
import numpy as np

images = []
labels = []
label_map = {}
data_dir = "dataset"

for idx, user in enumerate(os.listdir(data_dir)):
    user_dir = os.path.join(data_dir, user)
    label_map[idx] = user
    for img_name in os.listdir(user_dir):
        img_path = os.path.join(user_dir, img_name)
        features = preprocess_image(img_path)
        images.append(features)
        labels.append(idx)

X = np.array(images)
y = np.array(labels)

X_pca, pca_model = apply_pca(X)
X_lda, lda_model = apply_lda(X_pca, y)

svm_model = train_svm(X_lda, y)

test_image = preprocess_image("test_face.jpg")
test_pca = pca_model.transform([test_image])
test_lda = lda_model.transform(test_pca)
pred, conf = predict_svm(svm_model, test_lda)
print(f"Identified as: {label_map[pred[0]]} with confidence {conf[0][pred[0]]:.2f}")