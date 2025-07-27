from scapy.all import sniff, IP, TCP, get_if_list
import torch
import numpy as np
import pandas as pd
import joblib
import time
from collections import defaultdict
import threading
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# === Load Model, Scaler, and Feature List ===
model = torch.jit.load("ids_cnn_edgegensec.pt", map_location=torch.device("cpu"))
model.eval()

scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features_list.pkl")  # Contains original feature names with spaces

# === Flow Tracking Structures ===
flows = defaultdict(list)
flow_timestamps = {}
FLOW_TIMEOUT = 10  # seconds

# === Flow ID Generator ===
def get_flow_id(pkt):
    if IP in pkt and TCP in pkt:
        return (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, pkt[IP].proto)
    return None

# === Feature Extractor (Dummy for structure, needs CICIDS-style mapping for full value) ===
def extract_flow_features(packets):
    if len(packets) < 2:
        return [0.0] * len(feature_names)

    total_pkts = len(packets)
    total_bytes = sum(len(p) for p in packets)
    durations = packets[-1].time - packets[0].time

    sizes = [len(p) for p in packets]
    min_size = min(sizes)
    max_size = max(sizes)
    mean_size = sum(sizes) / total_pkts

    fwd_pkts = [p for p in packets if IP in p and p[IP].src.startswith("192.168")]
    bwd_pkts = [p for p in packets if IP in p and p[IP].dst.startswith("192.168")]

    basic_features = [
        durations,
        total_pkts,
        total_bytes,
        min_size,
        max_size,
        mean_size,
        len(fwd_pkts),
        len(bwd_pkts)
    ]

    padded_features = basic_features + [0.0] * (len(feature_names) - len(basic_features))
    return padded_features

# === Prediction Logic ===
def predict(features):
    try:
        df = pd.DataFrame([features], columns=feature_names)
        scaled = scaler.transform(df)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1)  # Shape: (1, 1, 78)
        with torch.no_grad():
            output = model(x)
            score = output.item()
            label = "Attack" if score > 0.5 else "Normal"
            logging.info(f"📡 Flow Detected: {label} (Score: {score:.4f})")
    except Exception as e:
        logging.warning(f"Prediction failed: {e}")

# === Flow Expiry Logic ===
def expire_flows():
    current_time = time.time()
    expired = []
    for fid, last_seen in flow_timestamps.items():
        if current_time - last_seen > FLOW_TIMEOUT:
            expired.append(fid)
    for fid in expired:
        packets = flows.pop(fid, [])
        flow_timestamps.pop(fid, None)
        if len(packets) >= 3:
            features = extract_flow_features(packets)
            predict(features)

def flow_expiry_loop():
    while True:
        time.sleep(2)
        expire_flows()

# === Packet Processor ===
def process_packet(pkt):
    fid = get_flow_id(pkt)
    if fid:
        flows[fid].append(pkt)
        flow_timestamps[fid] = time.time()

# === Main Entry Point ===
if __name__ == "__main__":
    print("🚨 Real-time flow-based IDS started...")
    print("Available interfaces:", get_if_list())

    iface = "wlan0"  # Update if needed for Raspberry Pi
    threading.Thread(target=flow_expiry_loop, daemon=True).start()
    sniff(filter="tcp", iface=iface, prn=process_packet, store=0)
