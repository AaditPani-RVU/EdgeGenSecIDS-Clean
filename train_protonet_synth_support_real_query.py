import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import os

# === 1. Encoder Definition ===
class Encoder(nn.Module):
    def __init__(self, input_dim=78, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# === 2. Few-Shot Sampling Utilities ===
def sample_episode(support_df, query_df, n_way=4, k_shot=5, q_query=10):
    classes = support_df["Label"].unique()
    selected_classes = np.random.choice(classes, n_way, replace=False)

    support_x, support_y = [], []
    query_x, query_y = [], []

    label_map = {label: i for i, label in enumerate(selected_classes)}

    for cls in selected_classes:
        s_cls = support_df[support_df["Label"] == cls]
        q_cls = query_df[query_df["Label"] == cls]

        s_samples = s_cls.sample(n=min(k_shot, len(s_cls)), replace=True)
        q_samples = q_cls.sample(n=min(q_query, len(q_cls)), replace=True)

        support_x.append(s_samples.drop(columns=["Label"]).values)
        support_y.extend([label_map[cls]] * len(s_samples))

        query_x.append(q_samples.drop(columns=["Label"]).values)
        query_y.extend([label_map[cls]] * len(q_samples))

    support_x = torch.tensor(np.vstack(support_x), dtype=torch.float32)
    query_x = torch.tensor(np.vstack(query_x), dtype=torch.float32)
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_y = torch.tensor(query_y, dtype=torch.long)

    return support_x, support_y, query_x, query_y

# === 3. ProtoNet Loss ===
def prototypical_loss(encoder, support_x, support_y, query_x, query_y):
    encoder.train()
    emb_support = encoder(support_x)
    emb_query = encoder(query_x)

    prototypes = []
    for cls in torch.unique(support_y):
        cls_indices = (support_y == cls).nonzero(as_tuple=True)[0]
        proto = emb_support[cls_indices].mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)

    logits = -torch.cdist(emb_query, prototypes)
    loss = F.cross_entropy(logits, query_y)
    acc = (logits.argmax(dim=1) == query_y).float().mean().item()
    return loss, acc

# === 4. Main Training Loop ===
if __name__ == "__main__":
    encoder = Encoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-4)

    df_synth = pd.read_csv("DATA/RARE_CLASSES/fewshot_synthetic_eval.csv")
    df_real = pd.read_csv("DATA/RARE_CLASSES/fewshot_real_eval.csv")

    episodes = 100
    best_acc = 0.0

    for ep in range(episodes):
        support_x, support_y, query_x, query_y = sample_episode(df_synth, df_real)
        loss, acc = prototypical_loss(encoder, support_x, support_y, query_x, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Episode {ep}] Accuracy: {acc:.4f} | Loss: {loss.item():.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(encoder.state_dict(), "protonet_encoder.pt")
            print("✅ Saved best encoder as protonet_encoder.pt")
