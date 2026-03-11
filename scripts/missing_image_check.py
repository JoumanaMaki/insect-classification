from pathlib import Path
import pandas as pd

df = pd.read_csv("data/ip102/metadata.csv")
img_dir = Path("data/ip102/images")

missing = df[~df["image"].map(lambda x: (img_dir / x).exists())]
print("Missing images:", len(missing))