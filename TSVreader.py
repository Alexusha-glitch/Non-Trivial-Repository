import pandas as pd
import os

def get_seizure_onset(tsv_path):
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        if df.empty:
            print(f"⚠️ TSV file is empty: {tsv_path}")
            return None

        row = df.iloc[0]

        vigilance = str(row.get("vigilance", "")).strip().lower()
        if vigilance in ["asleep"]:
            print(f"⚠️ Skipping due to vigilance = {vigilance} in {tsv_path}")
            return None

        onset = row.get("onset")
        if pd.isna(onset):
            print(f"⚠️ Onset missing in {tsv_path}")
            return None

        return float(onset)

    except Exception as e:
        print(f"⚠️ Error reading TSV {tsv_path}: {e}")
        return None
    
def find_matching_tsv(edf_path):
    base_dir="ds005873"
    edf_file = os.path.basename(edf_path)

    base_name = edf_file.replace("_ecg.edf", "")
    parts = base_name.split("_")

    sub_ses = f"{parts[0]}_{parts[1]}"
    tsv_dir = os.path.join(base_dir, f"{sub_ses}_tsv-files")
    tsv_file = f"{base_name}_events.tsv"
    tsv_path = os.path.join(tsv_dir, tsv_file)

    return tsv_path if os.path.exists(tsv_path) else None

def find_matching_tsv_emg(edf_path):
    base_dir="ds005873"
    edf_file = os.path.basename(edf_path)

    base_name = edf_file.replace("_emg.edf", "")
    parts = base_name.split("_")

    sub_ses = f"{parts[0]}_{parts[1]}"
    tsv_dir = os.path.join(base_dir, f"{sub_ses}_tsv-files")
    tsv_file = f"{base_name}_events.tsv"
    tsv_path = os.path.join(tsv_dir, tsv_file)

    return tsv_path if os.path.exists(tsv_path) else None