import os, io, urllib.request
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
OUTPUT = "data/heart.csv"

os.makedirs("data", exist_ok=True)

print(f"Downloading Heart Disease UCI dataset from:\n  {URL}")
with urllib.request.urlopen(URL) as resp:
    raw = resp.read().decode()

df = pd.read_csv(io.StringIO(raw), header=None, names=COLS, na_values="?")
df["target"] = (df["target"] > 0).astype(int)
df.to_csv(OUTPUT, index=False)

print(f"✔ Dataset saved to {OUTPUT}")
print(f"   Shape: {df.shape}")
print(f"   Missing values: {dict(df.isnull().sum()[df.isnull().sum() > 0])}")
print(f"   Class distribution: {dict(df['target'].value_counts())}")
