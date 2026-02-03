import numpy as np
import pandas as pd
import time

# Function to generate normal traffic with specific IP ranges
def generate_normal_traffic(num_samples):
    np.random.seed(42)
    lengths = np.random.randint(40, 1500, size=num_samples)  # Normal packet sizes
    protocols = np.random.choice([6, 17, 1], size=num_samples)  # TCP, UDP, ICMP
    ttls = np.random.randint(32, 128, size=num_samples)  # Normal TTL values
    src_ports = np.random.randint(1024, 65535, size=num_samples)
    dst_ports = np.random.randint(1024, 65535, size=num_samples)
    tcp_flags = np.random.choice(['SYN', 'ACK', 'PSH', 'RST'], size=num_samples)
    times = np.cumsum(np.random.exponential(scale=1, size=num_samples))

    # Normal traffic IP ranges: 192.168.x.x or 10.x.x.x
    sources = [f"192.168.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}" for _ in range(num_samples)]
    destinations = [f"10.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}" for _ in range(num_samples)]

    data = {
        'Time': times,
        'Source': sources,
        'Destination': destinations,
        'Length': lengths,
        'Protocol': protocols,
        'TTL': ttls,
        'Source Port': src_ports,
        'Destination Port': dst_ports,
        'TCP Flags': [tcp_flags[i] if protocols[i] == 6 else 'SYN' for i in range(num_samples)],
        'Info': [f'TCP Port {src_ports[i]} > {dst_ports[i]} [Data]' if protocols[i] == 6 else
                 f'UDP Port {src_ports[i]} > {dst_ports[i]} [Request]' if protocols[i] == 17 else
                 'ICMP Echo Request' for i in range(num_samples)]
    }

    return pd.DataFrame(data)

# Function to generate anomalous traffic with specific IP ranges
def generate_anomalous_traffic(num_samples):
    np.random.seed(42)
    lengths = np.random.choice([2000, 3000, 5000], size=num_samples)  # Larger packet sizes
    protocols = np.random.choice([6, 17, 1], size=num_samples)
    ttls = np.random.choice([1, 2, 5, 250], size=num_samples)  # Unusually low TTL values
    src_ports = np.random.randint(1, 1024, size=num_samples)  # Use reserved ports for anomalies
    dst_ports = np.random.randint(1, 1024, size=num_samples)  # Use reserved ports for anomalies
    tcp_flags = np.random.choice(['FIN', 'URG', 'RST'], size=num_samples)  # Uncommon flags
    times = np.cumsum(np.random.exponential(scale=1, size=num_samples))

    # Anomalous traffic IP ranges: 172.16.x.x - 172.31.x.x
    sources = [f"172.{np.random.randint(16, 32)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}" for _ in range(num_samples)]
    destinations = [f"172.{np.random.randint(16, 32)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}" for _ in range(num_samples)]

    data = {
        'Time': times,
        'Source': sources,
        'Destination': destinations,
        'Length': lengths,
        'Protocol': protocols,
        'TTL': ttls,
        'Source Port': src_ports,
        'Destination Port': dst_ports,
        'TCP Flags': [tcp_flags[i] if protocols[i] == 6 else 'SYN' for i in range(num_samples)],
        'Info': [f'TCP Port {src_ports[i]} > {dst_ports[i]} [Data]' if protocols[i] == 6 else
                 f'UDP Port {src_ports[i]} > {dst_ports[i]} [Request]' if protocols[i] == 17 else
                 'ICMP Echo Request' for i in range(num_samples)]
    }

    return pd.DataFrame(data)

# Define number of samples for normal and anomalous traffic
num_normal_samples = 100  # Increase normal samples
num_anomalous_samples = 20  # Fewer anomalous samples

# Generate normal and anomalous traffic
normal_traffic = generate_normal_traffic(num_normal_samples)
anomalous_traffic = generate_anomalous_traffic(num_anomalous_samples)

# Combine normal and anomalous traffic into a single dataframe
df = pd.concat([normal_traffic, anomalous_traffic], ignore_index=True)

# Add Time Delta feature based on the time difference between consecutive packets
df['Time Delta'] = df['Time'].diff().fillna(0)

# Save the combined dataset to a CSV file
df.to_csv('simulated_network_traffic_with_ip_ranges.csv', index=False)
print("Simulated dataset with specified IP ranges for normal and anomalous traffic saved as 'simulated_network_traffic_with_ip_ranges.csv'.")
