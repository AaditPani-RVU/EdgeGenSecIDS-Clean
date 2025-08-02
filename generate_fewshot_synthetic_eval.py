import os
import pandas as pd

rare_classes = ["SQLi", "XSS", "Heartbleed", "Infiltration"]
eval_samples_per_class = 100
output_path = "DATA/RARE_CLASSES/fewshot_synthetic_eval.csv"

dfs = []

for cls in rare_classes:
    path = f"DATA/RARE_CLASSES/{cls}_synthetic.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["Label"] = cls
        df_sample = df.sample(n=eval_samples_per_class, replace=len(df) < eval_samples_per_class, random_state=42)
        dfs.append(df_sample)
        print(f"✅ Sampled {len(df_sample)} for {cls}")
    else:
        print(f"⚠️ File not found: {path}")

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv(output_path, index=False)
print(f"✅ Saved: {output_path}")
