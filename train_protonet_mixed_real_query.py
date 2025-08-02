import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# === 1. Load Real Queries and Synthetic Support ===
df_real = pd.read_csv("DATA/RARE_CLASSES/fewshot_real_eval.csv")
df_synth = pd.read_csv("DATA/RARE_CLASSES/fewshot_synthetic_eval.csv")

# Ensure feature alignment
feature_cols = [col for col in df_real.columns if col != "Label"]
assert feature_cols == [col for col in df_synth.columns if col != "Label"]

# Label mapping
all_labels = sorted(set(df_real["Label"].unique()) | set(df_synth["Label"].unique()))
label_map = {label: i for i, label in enumerate(all_labels)}
df_real["Label"] = df_real["Label"].map(label_map)
df_synth["Label"] = df_synth["Label"].map(label_map)

# Normalize
scaler = StandardScaler()
df_synth[feature_cols] = scaler.fit_transform(df_synth[feature_cols])
df_real[feature_cols] = scaler.transform(df_real[feature_cols])

# === 2. Encoder (Deeper + Dropout) ===
class Encoder(nn.Module):
    def __init__(self, input_dim=78, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# === 3. Sample Mixed Episode ===
def sample_episode(real_df, synth_df, n_way=4, k_shot=5, q_query=10):
    support_x, support_y, query_x, query_y = [], [], [], []
    selected_classes = np.random.choice(real_df["Label"].unique(), n_way, replace=False)

    for cls in selected_classes:
        real_cls = real_df[real_df["Label"] == cls]
        synth_cls = synth_df[synth_df["Label"] == cls]

        s_samples = synth_cls.sample(n=k_shot, replace=len(synth_cls) < k_shot)
        q_samples = real_cls.sample(n=q_query, replace=len(real_cls) < q_query)

        support_x.append(s_samples[feature_cols].values)
        support_y.extend([cls] * k_shot)
        query_x.append(q_samples[feature_cols].values)
        query_y.extend([cls] * q_query)

    return (
        torch.tensor(np.vstack(support_x), dtype=torch.float32),
        torch.tensor(support_y, dtype=torch.long),
        torch.tensor(np.vstack(query_x), dtype=torch.float32),
        torch.tensor(query_y, dtype=torch.long)
    )

# === 4. ProtoNet Loss ===
def prototypical_loss(encoder, support_x, support_y, query_x, query_y):
    emb_support = encoder(support_x)
    emb_query = encoder(query_x)
    prototypes = torch.stack([emb_support[support_y == c].mean(0) for c in torch.unique(support_y)])
    logits = -torch.cdist(emb_query, prototypes)
    loss = F.cross_entropy(logits, query_y)
    acc = (logits.argmax(dim=1) == query_y).float().mean().item()
    return loss, acc

# === 5. Training Loop ===
encoder = Encoder()
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 regularization

best_acc = 0.0
for episode in range(100):
    support_x, support_y, query_x, query_y = sample_episode(df_real, df_synth)
    encoder.train()
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

print("✅ Saved best encoder as protonet_encoder.pt")
