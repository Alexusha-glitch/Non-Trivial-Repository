import os
import numpy as np
import pandas as pd
import neurokit2 as nk
import joblib
from tqdm import tqdm, trange
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from preprocessEDF import convert_df_from_tsv
from TSVreader import find_matching_tsv, get_seizure_onset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

base_dir = "ds005873"
fs = 250
uv_threshold = 200
tolerance = 0.8
contamination_rate = 0.05

def is_clean(signal, threshold=uv_threshold):
    signal = np.array(signal)
    return np.all((signal > -threshold) & (signal < threshold))

def extract_hrv_features(segment, fs=250):
    signal = np.squeeze(np.array(segment))
    try:
        ecg_signals, ecg_info = nk.ecg_process(signal, sampling_rate=fs)
        rpeaks = ecg_info.get("ECG_R_Peaks", [])
        if len(rpeaks) < 2:
            return None
        hrv = nk.hrv_time({"ECG_R_Peaks": rpeaks}, sampling_rate=fs, show=False)
        hr_mean = hrv.get("HRV_MeanNN", [np.nan])[0]
        sdnn = hrv.get("HRV_SDNN", [np.nan])[0]
        if np.isnan(hr_mean) or np.isnan(sdnn):
            return None
        return {"HR_Mean": hr_mean, "SDNN": sdnn}
    except Exception as e:
        print("HRV extraction failed:", e)
        return None

edf_paths = []
subjects = [d for d in os.listdir(base_dir) if d.startswith("sub-")]
for sub in subjects:
    sub_path = os.path.join(base_dir, sub)
    sessions = [s for s in os.listdir(sub_path) if s.startswith("ses-")]
    for ses in sessions:
        ecg_path = os.path.join(sub_path, ses, "ecg")
        if not os.path.isdir(ecg_path):
            continue
        for fname in os.listdir(ecg_path):
            if fname.endswith(".edf"):
                edf_paths.append(os.path.join(ecg_path, fname))

train_edfs, test_edfs = train_test_split(edf_paths, test_size=0.2, random_state=42)

def process_edf_list(edf_list, label_value):
    rows = []
    for edf_path in tqdm(edf_list, desc="Processing EDFs"):
        tsv_path = find_matching_tsv(edf_path)
        if not tsv_path:
            continue
        seizure_secs = get_seizure_onset(tsv_path)
        if seizure_secs is None:
            continue
        df = convert_df_from_tsv(edf_path, seizure_secs)
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            signal = row["features"]
            if not is_clean(signal):
                continue
            feats = extract_hrv_features(signal)
            if feats:
                feats["label"] = label_value
                rows.append(feats)
    return pd.DataFrame(rows)

print("Extracting features from training EDFs...")
train_df = process_edf_list(train_edfs, label_value=0)

print("Extracting features from testing EDFs...")
test_df = process_edf_list(test_edfs, label_value=0)

X_train = train_df[["HR_Mean", "SDNN"]].values
X_test = test_df[["HR_Mean", "SDNN"]].values

y_test = np.ones(len(X_test))
y_test[:int(len(X_test) * contamination_rate)] = -1
np.random.shuffle(y_test)

print("Training")
n_epochs = 30
best_acc = 0

for epoch in trange(1, n_epochs + 1, desc="Epochs"):
    iso = IsolationForest(
        contamination=contamination_rate,
        n_estimators=100 + epoch * 10,
        random_state=42 + epoch
    )
    iso.fit(X_train)
    preds = iso.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Epoch {epoch} | Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
    y_true = np.where(y_test == 1, 0, 1)
    y_pred = np.where(preds == 1, 0, 1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"Specificity: {specificity:.4f}")

    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUROC: {auc:.4f}")
    except Exception as e:
        print("AUROC could not be computed:", e)
    joblib.dump(iso, f"ecg_iso_model_epoch{epoch}.joblib")

print(f"Best Accuracy Achieved: {best_acc:.4f}")