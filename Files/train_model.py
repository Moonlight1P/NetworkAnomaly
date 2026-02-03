import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import time

class AnomalyDetector:
    def __init__(self, contamination=0.03):  # Adjusted contamination rate should be in a float in the range (0.0, 0.5]
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.last_packet_time = None

    def extract_features(self, row):
        current_time = time.time()
        time_delta = current_time - self.last_packet_time if self.last_packet_time is not None else 0
        self.last_packet_time = current_time

        features = [
            row['Length'],
            row['Protocol'],
            row['TTL'],
            time_delta,
            row['Source Port'],
            row['Destination Port'],
            row['TCP Flags']
        ]
        return features

    def preprocess_features(self, X):
        X_normalized = self.scaler.fit_transform(X)
        return X_normalized

    def train(self, X_train):
        X_train_processed = self.preprocess_features(X_train)
        self.model.fit(X_train_processed)

    def save_model(self, model_path, scaler_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError("CSV file is empty or not loaded correctly.")

    print("Sample of loaded data:")
    print(df.head())

    df['Protocol'] = df['Info'].apply(lambda x: 6 if 'TCP' in x else (17 if 'UDP' in x else (1 if 'ICMP' in x else 0)))
    flag_mapping = {'SYN': 1, 'ACK': 2, 'PSH': 3, 'RST': 4, 'FIN': 5, 'URG': 6, 'N/A': 0}
    df['TCP Flags'] = df['TCP Flags'].map(flag_mapping)

    columns_to_convert = ['Length', 'TTL', 'Source Port', 'Destination Port', 'TCP Flags']
    for column in columns_to_convert:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    print("NaNs in each column after conversion:")
    print(df[columns_to_convert].isna().sum())

    df.dropna(subset=columns_to_convert, inplace=True)
    if df.empty:
        raise ValueError("No valid data available after preprocessing.")

    print(f"Shape of processed data: {df.shape}")
    return df

if __name__ == "__main__":
    csv_file_path = 'simulated_network_traffic3.csv'
    df = load_data_from_csv(csv_file_path)

    detector = AnomalyDetector()

    # Ensure features is a 2D array
    features = np.array([detector.extract_features(row) for _, row in df.iterrows()])

    # Confirm shape before training
    print(f"Features shape: {features.shape}")

    detector.train(features)

    detector.save_model('anomaly_detector_model3.pkl', 'scaler3.pkl')
    print("Model and scaler saved successfully.")
