import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

DATA = Path("data/ip102")
df = pd.read_csv(DATA / "metadata.csv").sample(9, random_state=42)

fig, axs = plt.subplots(3, 3, figsize=(9, 9))

for ax, (_, row) in zip(axs.flatten(), df.iterrows()):
    img = Image.open(DATA / "images" / row["image"]).convert("RGB")
    ax.imshow(img)
    ax.set_title(f"label={row['label']}")
    ax.axis("off")

plt.tight_layout()
plt.show()