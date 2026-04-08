import pandas as pd
import glob
import re
import os

# 1. DYNAMIC PATH SELECTION
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# Path: root -> Data -> Traduction avis clients
data_folder = os.path.join(project_root, "Data", "Traduction avis clients")
all_files = glob.glob(os.path.join(data_folder, "*.xlsx"))

if not all_files:
    print(f"Error: No Excel files found in {data_folder}!")
else:
    print(f"Merging {len(all_files)} files...")
    df = pd.concat([pd.read_excel(f) for f in all_files], ignore_index=True)

    # 2. FAST CLEANING (Skips slow spelling correction for now)
    def fast_clean(text):
        if pd.isna(text): return ""
        # Requirement: Data cleaning/standardization [cite: 67]
        text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
        return text.strip()

    print("Cleaning text (Fast Mode)...")
    # Using 'avis' as the source column
    df['cleaned_text'] = df['avis'].apply(fast_clean)
    
    # 3. IMMEDIATE SAVE
    df = df[df['cleaned_text'] != ""] # Ensure no empty rows for the model
    output_path = os.path.join(project_root, "processed_data.csv")
    
    # SAVE NOW so the file actually appears
    df.to_csv(output_path, index=False)
    print(f"SUCCESS! {output_path} created. You can now run train_model.py.")