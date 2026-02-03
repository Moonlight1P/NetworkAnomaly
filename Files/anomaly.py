import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=42)
    
    def train(self, X_train):
        X_train_normalized = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_normalized)
    
    def predict(self, features):
        feature_vector_normalized = self.scaler.transform([features])
        prediction = self.model.predict(feature_vector_normalized)
        return prediction[0]  # -1 indicates an anomaly, 1 indicates normal
    
    def is_anomaly(self, features):
        return self.predict(features) == -1


# Example usage (this would be removed when using it as an imported module)
if __name__ == "__main__":
    # Simulated training data (replace with actual data)
    X_train = np.array([
        [10, 100, 64, 0], [12, 110, 63, 0], [15, 90, 60, 0], [11, 105, 61, 0], [14, 95, 62, 0],  # Normal traffic examples
        [1500, 6, 128, 0], [2000, 6, 255, 0], [3000, 17, 128, 0], [5000, 1, 64, 0], [10000, 6, 64, 0],  # Large packets
        [40, 6, 128, 0], [50, 6, 255, 0], [60, 17, 64, 0], [80, 1, 63, 0],  # High frequency small packets
    ])

    detector = AnomalyDetector()
    detector.train(X_train)

    # Test with a sample feature vector
    sample_features = [500, 50, 10, 1]  # Replace with actual features
    print(detector.is_anomaly(sample_features))
