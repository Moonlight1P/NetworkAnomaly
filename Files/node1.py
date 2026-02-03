from scapy.all import sniff
from anomaly import AnomalyDetector
import numpy as np

# Initialize the anomaly detector
detector = AnomalyDetector(contamination=0.1)

# Train the detector (replace with actual training data)
X_train = np.array([
    [10, 100, 64, 0], [12, 110, 63, 0], [15, 90, 60, 0], [11, 105, 61, 0], [14, 95, 62, 0],  # Normal traffic examples
    [1500, 6, 128, 0], [2000, 6, 255, 0], [3000, 17, 128, 0], [5000, 1, 64, 0], [10000, 6, 64, 0],  # Large packets
    [40, 6, 128, 0], [50, 6, 255, 0], [60, 17, 64, 0], [80, 1, 63, 0],  # High frequency small packets
])
detector.train(X_train)


# Feature Extraction
def extract_features(packet):
    try:
        ip_layer = packet['IP']
        if packet.haslayer('TCP'):
            transport_layer = packet['TCP']
        elif packet.haslayer('UDP'):
            transport_layer = packet['UDP']
        elif packet.haslayer('ICMP'):
            transport_layer = packet['ICMP']
        else:
            transport_layer = None

        features = [
            len(packet),                   # Packet length
            ip_layer.proto,                # Protocol
            ip_layer.ttl,                  # Time to live
            int(transport_layer.flags) if hasattr(transport_layer, 'flags') else 0,  # Flags (for TCP)
        ]
        return features
    except IndexError:
        return None


# Real-time packet processing and anomaly detection
def packet_callback(packet):
    features = extract_features(packet)
    if features:
        if detector.is_anomaly(features):
            print(f"Anomaly detected in packet: {packet.summary()}")
        else:
            print(f"Normal traffic: {packet.summary()}")


# Start sniffing and real-time monitoring
print("Capturing traffic: ")
sniff(prn=packet_callback, store=0)
