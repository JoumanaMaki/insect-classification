from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class IP102Dataset(Dataset):
    def __init__(self, metadata_csv, images_dir, split='train', transform=None):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["image"]
        image = Image.open(img_path).convert('RGB')
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label