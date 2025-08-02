import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os

# === 1. Encoder Definition ===
# === Encoder Definition (Deeper MLP with dropout) ===
# === Encoder Definition (Match trained model: hidden_dim=64) ===
# Updated encoder architecture (3-layer MLP with 128 units)
class Encoder(nn.Module):
    def __init__(self, input_dim=78, hidden_dim=128, output_dim=128):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)



# === 2. Evaluation Logic ===
def evaluate_encoder(encoder, support_x, support_y, query_x, query_y, batch_size=256):
    encoder.eval()
    total, correct = 0, 0
    with torch.no_grad():
        emb_support = encoder(support_x)
        prototypes = torch.stack([emb_support[support_y == c].mean(0) for c in torch.unique(support_y)])
        for i in range(0, len(query_x), batch_size):
            batch_x = query_x[i:i+batch_size]
            batch_y = query_y[i:i+batch_size]
            emb_query = encoder(batch_x)
            logits = -torch.cdist(emb_query, prototypes)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total

# === 3. Load Real Queries ===
rare_classes = ["SQLi", "XSS", "Heartbleed", "Infiltration"]
real_query_df = []

for cls in rare_classes:
    path = f"DATA/RARE_CLASSES/{cls}_real.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["Label"] = cls
        real_query_df.append(df)

df = pd.concat(real_query_df, ignore_index=True)
feature_cols = [col for col in df.columns if col != "Label"]
label_map = {name: i for i, name in enumerate(sorted(df["Label"].unique()))}
df["Label"] = df["Label"].map(label_map)

X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
y = torch.tensor(df["Label"].values, dtype=torch.long)

# === 4. Load Support Set + Encoder ===
support_x = torch.load("DATA/RARE_CLASSES/support_x.pt")
support_y = torch.load("DATA/RARE_CLASSES/support_y.pt")

encoder = Encoder()
encoder.load_state_dict(torch.load("protonet_encoder.pt"))
encoder.eval()

# === 5. Evaluate ===
acc = evaluate_encoder(encoder, support_x, support_y, X, y)
print(f"✅ Final Few-Shot Accuracy on Real Queries: {acc:.4f}")
