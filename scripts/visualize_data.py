import os
import pandas as pd
import matplotlib.pyplot as plt

# === Paths ===
metadata_path = r"C:/Users/Bouchra/Med_AI_Sys/data/metadata/metadata.csv"
images_folder = r"C:/Users/Bouchra/Med_AI_Sys/data/Images/"
plots_folder = r"C:/Users/Bouchra/Med_AI_Sys/data/plots/"

# === Ensure output folder exists ===
os.makedirs(plots_folder, exist_ok=True)

# === Load metadata ===
metadata = pd.read_csv(metadata_path)
metadata['image_file'] = metadata['isic_id'].apply(lambda x: x + '.jpg')
metadata['image_path'] = metadata['image_file'].apply(lambda f: os.path.join(images_folder, f))
metadata = metadata[metadata['image_path'].apply(os.path.exists)].reset_index(drop=True)

print(f" Loaded {len(metadata)} valid image entries.\n")

# === Example 1: Distribution of primary diagnosis ===
plt.figure(figsize=(10, 6))
metadata['diagnosis_1'].value_counts().plot(kind='bar', color='teal')
plt.title("Distribution of Primary Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')

# Save + show
save_path = os.path.join(plots_folder, "diagnosis_distribution.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f" Saved plot: {save_path}")
plt.show()

# === Example 2: Age distribution (if available) ===
if 'age_approx' in metadata.columns:
    plt.figure(figsize=(8, 5))
    metadata['age_approx'].dropna().plot(kind='hist', bins=30, color='orange', edgecolor='black')
    plt.title("Age Distribution of Patients")
    plt.xlabel("Approximate Age")
    plt.ylabel("Frequency")

    save_path = os.path.join(plots_folder, "age_distribution.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f" Saved plot: {save_path}")
    plt.show()
