import pandas as pd
import os

# Path to real data directory
real_dir = "DATA/RARE_CLASSES"

# Output CSV
output_csv = os.path.join(real_dir, "fewshot_real_eval.csv")

# Labels and samples per class
labels = ["SQLi", "XSS", "Heartbleed", "Infiltration"]
samples_per_class = 100

dfs = []
for label in labels:
    file_path = os.path.join(real_dir, f"{label}_real.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.dropna()
        df = df.sample(n=min(samples_per_class, len(df)), random_state=42)
        df["Label"] = label
        dfs.append(df)
    else:
        print(f"⚠️ Missing: {file_path}")

# Combine and save
df_final = pd.concat(dfs, ignore_index=True)
df_final.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv} ({df_final.shape})")
