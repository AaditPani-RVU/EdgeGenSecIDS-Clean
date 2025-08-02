import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# === 1. Encoder Definition ===
class Encoder(nn.Module):
    def __init__(self, input_dim=78, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

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

# === 3. Run Evaluation ===
if __name__ == "__main__":
    encoder = Encoder()
    encoder.load_state_dict(torch.load(r"C:\Users\aadip\Desktop\internship\Edgegensec_clean\protonet_encoder.pt"))
    encoder.eval()

    # === Load and preprocess evaluation data ===
    df = pd.read_csv("DATA/RARE_CLASSES/fewshot_synthetic_eval.csv")
    feature_cols = [col for col in df.columns if col != "Label"]

    # Map string labels to integers using the same label_map used in training
    label_map = {"SQLi": 0, "XSS": 1, "Heartbleed": 2, "Infiltration": 3}
    df["Label"] = df["Label"].map(label_map)

    # Convert to tensors
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["Label"].values, dtype=torch.long)

    # Load support set (must match same class order and label mapping)
    support_x = torch.load("DATA/RARE_CLASSES/support_x.pt")
    support_y = torch.load("DATA/RARE_CLASSES/support_y.pt")

    acc = evaluate_encoder(encoder, support_x, support_y, X, y)
    print(f"✅ Final Few-Shot Accuracy on Synthetic Queries: {acc:.4f}")
