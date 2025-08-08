from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

def train_knn_model(X_train, y_train, n_neighbors=5):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn_classifier.fit(X_train, y_train)
    dump(knn_classifier, 'knn_model.joblib')
    print(f"K-NN model with k={n_neighbors} trained and saved.")
    return knn_classifier
