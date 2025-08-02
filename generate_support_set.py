import pandas as pd
import numpy as np
import torch

# === Load the same few-shot evaluation CSV ===
df = pd.read_csv("DATA/RARE_CLASSES/fewshot_synthetic_eval.csv")

# === Define feature columns and label mapping ===
feature_cols = [col for col in df.columns if col != "Label"]
label_map = {"SQLi": 0, "XSS": 1, "Heartbleed": 2, "Infiltration": 3}
df["Label"] = df["Label"].map(label_map).astype(int)

# === Generate support set (5-shot per class) ===
support_x, support_y = [], []
for label_int in sorted(label_map.values()):
    class_df = df[df["Label"] == label_int]
    support_samples = class_df.sample(n=5, replace=False)
    support_x.append(support_samples[feature_cols].values)
    support_y.append([label_int] * 5)

support_x = torch.tensor(np.vstack(support_x), dtype=torch.float32)
support_y = torch.tensor(np.concatenate(support_y), dtype=torch.long)

# === Save to disk ===
torch.save(support_x, "DATA/RARE_CLASSES/support_x.pt")
torch.save(support_y, "DATA/RARE_CLASSES/support_y.pt")
print("✅ Saved support_x.pt and support_y.pt")
