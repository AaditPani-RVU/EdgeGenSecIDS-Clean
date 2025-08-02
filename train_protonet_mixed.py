import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# === Load Real + Synthetic Data ===
rare_classes = ["SQLi", "XSS", "Heartbleed", "Infiltration"]
dfs = []

for cls in rare_classes:
    for kind in ["real", "synthetic"]:
        path = f"DATA/RARE_CLASSES/{cls}_{kind}.csv"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df_part = pd.read_csv(path)
            df_part.columns = [c.strip() for c in df_part.columns]
            df_part["Label"] = cls
            dfs.append(df_part)

if not dfs:
    raise ValueError("❌ No valid data found.")

df = pd.concat(dfs, ignore_index=True)
df.dropna(inplace=True)

# === Feature Columns ===
feature_cols = [col for col in df.columns if col != "Label"]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# === Label Encoding ===
unique_labels = sorted(df["Label"].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)
df["Label"] = df["Label"].map(label_map)

# === Save for inference reproducibility ===
import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "features_list.pkl")

# === Encoder ===
class Encoder(nn.Module):
    def __init__(self, input_dim=78, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.relu(self.drop1(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x

# === Episode Sampler ===
def sample_episode(df, n_way=4, k_shot=5, q_query=10):
    support_x, support_y, query_x, query_y = [], [], [], []
    selected_classes = sorted(df["Label"].unique())[:n_way]
    for cls in selected_classes:
        class_df = df[df["Label"] == cls]
        n_samples = k_shot + q_query
        replace = len(class_df) < n_samples
        samples = class_df.sample(n=n_samples, replace=replace)
        support = samples.iloc[:k_shot]
        query = samples.iloc[k_shot:]
        support_x.append(support[feature_cols].values)
        support_y.append([cls] * k_shot)
        query_x.append(query[feature_cols].values)
        query_y.append([cls] * q_query)

    return (
        torch.tensor(np.vstack(support_x), dtype=torch.float32),
        torch.tensor(np.concatenate(support_y), dtype=torch.long),
        torch.tensor(np.vstack(query_x), dtype=torch.float32),
        torch.tensor(np.concatenate(query_y), dtype=torch.long)
    )

# === Proto Loss ===
def prototypical_loss(encoder, support_x, support_y, query_x, query_y):
    emb_support = encoder(support_x)
    emb_query = encoder(query_x)
    prototypes = torch.stack([emb_support[support_y == c].mean(0) for c in torch.unique(support_y)])
    logits = -torch.cdist(emb_query, prototypes)
    loss = F.cross_entropy(logits, query_y)
    acc = (logits.argmax(dim=1) == query_y).float().mean().item()
    return loss, acc

# === Training Loop ===
encoder = Encoder(input_dim=len(feature_cols))
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

best_acc = 0
for episode in range(100):
    support_x, support_y, query_x, query_y = sample_episode(df, k_shot=5, q_query=15)
    optimizer.zero_grad()
    loss, acc = prototypical_loss(encoder, support_x, support_y, query_x, query_y)
    if torch.isnan(loss): continue
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        print(f"[Episode {episode}] Accuracy: {acc:.4f} | Loss: {loss.item():.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(encoder.state_dict(), "protonet_encoder.pt")

print("✅ Saved improved encoder as protonet_encoder.pt")
