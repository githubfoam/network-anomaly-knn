#### 4. `src/knn_model.py`

This file contains the Python code for the K-NN anomaly detection model.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class KnnAnomalyDetector:
    """
    K-Nearest Neighbors based Anomaly Detector.

    This model identifies anomalies by measuring the distance of each data point
    to its k-th nearest neighbor. Points with large distances are considered anomalies.
    """

    def __init__(self, n_neighbors=5, contamination=0.1):
        """
        Initializes the KnnAnomalyDetector.

        Args:
            n_neighbors (int): Number of neighbors to use for anomaly detection.
            contamination (float): The proportion of outliers in the data set.
        """
        if not 0.0 < contamination < 0.5:
            raise ValueError("Contamination must be in the range (0, 0.5).")
        
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = NearestNeighbors(n_neighbors=n_neighbors)
        self.threshold = None

    def fit(self, X):
        """
        Fits the NearestNeighbors model and determines the anomaly threshold.

        Args:
            X (np.ndarray): The input data.
        """
        self.model.fit(X)
        distances, _ = self.model.kneighbors(X)
        kth_distances = distances[:, -1]
        
        # Determine the anomaly threshold based on contamination
        self.threshold = np.percentile(kth_distances, 100 * (1 - self.contamination))

    def predict(self, X):
        """
        Predicts if a data point is an anomaly.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: A boolean array where True indicates an anomaly.
        """
        if self.threshold is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        distances, _ = self.model.kneighbors(X)
        kth_distances = distances[:, -1]
        
        return kth_distances > self.threshold

    def score_samples(self, X):
        """
        Calculates the anomaly score for each sample.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The anomaly scores (distance to the k-th nearest neighbor).
        """
        if self.threshold is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        distances, _ = self.model.kneighbors(X)
        return distances[:, -1]

# Example usage (for demonstration)
if __name__ == "__main__":
    # Sample data: one anomaly (10, 10)
    X_train = np.array([
        [1, 1], [1.5, 2], [2, 1.5], [8, 9], [9, 8], [9.5, 9.5],
        [1.1, 1.3], [1.8, 1.9], [8.5, 9.2], [9.1, 8.8], [10, 10]
    ])

    detector = KnnAnomalyDetector(n_neighbors=2, contamination=0.1)
    detector.fit(X_train)
    
    predictions = detector.predict(X_train)
    scores = detector.score_samples(X_train)
    
    print("Anomaly predictions (True = anomaly):")
    print(predictions)
    print("\nAnomaly scores:")
    print(scores)