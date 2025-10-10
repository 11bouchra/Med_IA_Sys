import pandas as pd
import os
"""
# Correct absolute paths
metadata_path = r"C:/Users/Bouchra/Med_AI_Sys/data/metadata/metadata.csv"
images_folder = r"C:/Users/Bouchra/Med_AI_Sys/data/Images/"

# Load metadata
metadata = pd.read_csv(metadata_path)

# Show overview
print("Metadata overview:")
print(metadata.head())

# Count image files
if os.path.exists(images_folder):
    num_images = len([f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print("\n Number of images in folder:", num_images)
else:
    print("\n Images folder not found at:", images_folder)

# Show missing values
print("\n Missing values per column:")
print(metadata.isnull().sum())

"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

metadata = pd.read_csv(r"C:/Users/Bouchra/Med_AI_Sys/data/metadata/metadata.csv")
images_folder = r"C:/Users/Bouchra/Med_AI_Sys/data/Images/"

# Make sure image_file column exists (from Step 1) or create using a simple convention
metadata['image_file'] = metadata['isic_id'].apply(lambda x: x + '.jpg')  # update if ext differs
metadata['image_path'] = metadata['image_file'].apply(lambda f: os.path.join(images_folder, f))
# Keep only rows whose image file exists
metadata = metadata[metadata['image_path'].apply(os.path.exists)].reset_index(drop=True)

# Stratified split by diagnosis_1 (change label column as needed)
train_df, temp_df = train_test_split(metadata, test_size=0.30, stratify=metadata['diagnosis_1'], random_state=42)
val_df, test_df  = train_test_split(temp_df, test_size=0.50, stratify=temp_df['diagnosis_1'], random_state=42)

os.makedirs('data/splits', exist_ok=True)
train_df.to_csv('data/splits/train.csv', index=False)
val_df.to_csv('data/splits/val.csv', index=False)
test_df.to_csv('data/splits/test.csv', index=False)
print("Saved train/val/test CSVs with counts:", len(train_df), len(val_df), len(test_df))

# PyTorch Dataset skeleton (classification)
"""
from PIL import Image
import torch
from torch.utils.data import Dataset

class ISICClassificationDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # convert label to numeric if necessary
        label = row['diagnosis_1']
        return img, label
"""