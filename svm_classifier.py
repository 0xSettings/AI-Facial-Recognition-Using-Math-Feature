from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(features, labels):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(features, labels)
    return clf

def predict_svm(clf, new_features):
    return clf.predict(new_features), clf.predict_proba(new_features)