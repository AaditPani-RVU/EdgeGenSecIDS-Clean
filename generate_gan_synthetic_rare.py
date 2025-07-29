import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# === GAN CONFIG ===
RARE_CLASSES = ["SQLi", "XSS", "Heartbleed", "Infiltration"]
LATENT_DIM = 32
EPOCHS = 5000
BATCH_SIZE = 16
OUTPUT_SAMPLES = 1000

# === MODELS ===
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# === TRAINING LOOP PER CLASS ===
for cls in RARE_CLASSES:
    real_path = f"DATA/RARE_CLASSES/{cls}_real.csv"
    syn_path = f"DATA/RARE_CLASSES/{cls}_synthetic.csv"

    if not os.path.exists(real_path):
        print(f"❌ Missing: {real_path}")
        continue

    df = pd.read_csv(real_path)
    feature_cols = [c for c in df.columns if c != "Label"]
    if not feature_cols:
        print(f"⚠️ Skipping {cls}, no features found.")
        continue

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    X = torch.tensor(X, dtype=torch.float32)

    G = Generator(LATENT_DIM, X.shape[1])
    D = Discriminator(X.shape[1])
    g_opt = optim.Adam(G.parameters(), lr=1e-3)
    d_opt = optim.Adam(D.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        real_idx = torch.randint(0, X.shape[0], (BATCH_SIZE,))
        real_samples = X[real_idx]
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        fake_samples = G(z).detach()

        d_real = D(real_samples)
        d_fake = D(fake_samples)
        d_loss = loss_fn(d_real, torch.ones_like(d_real)) + loss_fn(d_fake, torch.zeros_like(d_fake))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        fake_samples = G(z)
        d_gen = D(fake_samples)
        g_loss = loss_fn(d_gen, torch.ones_like(d_gen))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    with torch.no_grad():
        z = torch.randn(OUTPUT_SAMPLES, LATENT_DIM)
        syn_samples = G(z).numpy()
        syn_df = pd.DataFrame(scaler.inverse_transform(syn_samples), columns=feature_cols)
        syn_df.to_csv(syn_path, index=False)
        print(f"✅ {cls}: Saved {OUTPUT_SAMPLES} synthetic samples to {syn_path}")
