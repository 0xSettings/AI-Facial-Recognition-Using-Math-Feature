from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def apply_pca(data, n_components=50):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data), pca

def apply_lda(data, labels, n_components=10):
    lda = LDA(n_components=n_components)
    return lda.fit_transform(data, labels), lda