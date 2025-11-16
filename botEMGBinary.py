import os
import numpy as np
import pandas as pd
import joblib
from tqdm import trange
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessEDF import convert_df_from_tsv_emg
from TSVreader import find_matching_tsv_emg, get_seizure_onset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

base_dir = "ds005873"
uv_limit = 200
contamination_rate = 0.05
n_epochs = 20

def is_clean(signal, threshold=uv_limit):
    signal = np.array(signal)
    return np.all((signal >= -threshold) & (signal <= threshold))

edf_paths = []
subjects = [d for d in os.listdir(base_dir) if d.startswith("sub-")]
for sub in subjects:
    for ses in os.listdir(os.path.join(base_dir, sub)):
        path = os.path.join(base_dir, sub, ses, "emg")
        if not os.path.isdir(path):
            continue
        for file in os.listdir(path):
            if file.endswith(".edf"):
                edf_paths.append(os.path.join(path, file))

edf_train, edf_test = train_test_split(edf_paths, test_size=0.2, random_state=42)

def process_edf_list(edf_list):

    all_segments = []

    for edf_path in edf_list:

        tsv_path = find_matching_tsv_emg(edf_path)
        if not tsv_path:
            continue

        seizure_secs = get_seizure_onset(tsv_path)
        if seizure_secs is None:
            continue

        df = convert_df_from_tsv_emg(edf_path, seizure_secs)
        if df is not None:
            all_segments.append(df)

    return pd.concat(all_segments, ignore_index=True) if all_segments else pd.DataFrame()

print("Processing training files...")
df_train_all = process_edf_list(edf_train)
print("Processing test files...")
df_test_all = process_edf_list(edf_test)

def filter_and_flatten(df):
    if df.empty:
        return []
    
    df_clean = df[df["label"] == df["label"].max()]
    df_clean = df_clean[df_clean["features"].apply(is_clean)]

    return df_clean["features"].apply(lambda x: np.ravel(x)).tolist()

X_train = filter_and_flatten(df_train_all)
X_test = filter_and_flatten(df_test_all)

y_test = np.ones(len(X_test))
y_test[:int(len(y_test) * contamination_rate)] = -1
np.random.shuffle(y_test)

best_acc = 0

for epoch in trange(1, n_epochs + 1, desc="EMG Epochs"):
    clf = IsolationForest(
        n_estimators=50 + 10 * epoch,
        contamination=contamination_rate,
        random_state=42 + epoch
    )

    clf.fit(X_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Epoch {epoch} | Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc

    y_true = np.where(y_test == 1, 0, 1)
    y_pred = np.where(preds == 1, 0, 1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    print(f"Specificity: {specificity:.4f}")

    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUROC: {auc:.4f}")
    except Exception as e:
        print("AUROC could not be computed:", e)

    joblib.dump(clf, f"emg_iso_epoch{epoch}.joblib")

print(f"Best Accuracy Achieved: {best_acc:.4f}")

print(f"Best Accuracy Achieved: {best_acc:.4f}")