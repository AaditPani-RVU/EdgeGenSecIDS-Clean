import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from model import Encoder  # Ensure this matches your encoder class
import seaborn as sns

# === Load datasets ===
support_df = pd.read_csv("DATA/RARE_CLASSES/fewshot_train_mixed_support.csv")  # real+synthetic support
query_df = pd.read_csv("DATA/RARE_CLASSES/fewshot_real_eval.csv")              # real-only query

# === Encode labels ===
le = LabelEncoder()
all_labels = pd.concat([support_df["Label"], query_df["Label"]])
le.fit(all_labels)

support_y = le.transform(support_df["Label"])
query_y = le.transform(query_df["Label"])

support_x = support_df.drop(columns=["Label"]).values
query_x = query_df.drop(columns=["Label"]).values

# === Load encoder ===
encoder = Encoder()
encoder.load_state_dict(torch.load("protonet_encoder.pt"))
encoder.eval()

# === Embed samples ===
with torch.no_grad():
    emb_support = encoder(torch.tensor(support_x, dtype=torch.float32)).numpy()
    emb_query = encoder(torch.tensor(query_x, dtype=torch.float32)).numpy()

# === Combine for t-SNE ===
X = np.concatenate([emb_support, emb_query], axis=0)
y = np.concatenate([support_y, query_y], axis=0)
domains = ["Support"] * len(support_y) + ["Query"] * len(query_y)

# === t-SNE ===
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
X_tsne = tsne.fit_transform(X)

# === Plot ===
df_vis = pd.DataFrame({
    "x": X_tsne[:, 0],
    "y": X_tsne[:, 1],
    "Label": le.inverse_transform(y),
    "Domain": domains
})

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_vis, x="x", y="y", hue="Label", style="Domain", s=80, alpha=0.8)
plt.title("t-SNE of Support (synthetic+real) and Real Query Embeddings")
plt.grid(True)
plt.tight_layout()
plt.savefig("embedding_visualization_tsne.png", dpi=300)
plt.show()
