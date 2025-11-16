import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocessEDF import convert_df_from_tsv_emg
from TSVreader import find_matching_tsv_emg, get_seizure_onset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

def flatten_and_pool(arr_list, target_len=50):
    pooled = []
    for arr in arr_list:
        arr = np.array(arr)
        pooled_arr = []
        for ch in arr:
            pooled_ch = np.interp(np.linspace(0, len(ch), target_len), np.arange(len(ch)), ch)
            pooled_arr.extend(pooled_ch)
        pooled.append(pooled_arr)
    return pooled

base_dir = "ds005873"
all_segments = []

subjects = [d for d in os.listdir(base_dir) if d.startswith("sub-") and os.path.isdir(os.path.join(base_dir, d))]
for sub in tqdm(subjects, desc="🔍 Subjects"):
    sub_path = os.path.join(base_dir, sub)

    sessions = [s for s in os.listdir(sub_path) if s.startswith("ses-")]
    for ses in tqdm(sessions, desc=f"  📁 {sub}", leave=False):
        ses_path = os.path.join(sub_path, ses)
        emg_path = os.path.join(ses_path, "emg")

        if not os.path.isdir(emg_path):
            continue

        edf_files = [f for f in os.listdir(emg_path) if f.endswith(".edf")]
        for fname in tqdm(edf_files, desc=f"    📄 {sub}/{ses}", leave=False):
            edf_path = os.path.join(emg_path, fname)

            print(f"\n📄 Processing: {edf_path}")
            tsv_path = find_matching_tsv_emg(edf_path)

            if not tsv_path:
                print(f"❌ No TSV found for: {edf_path}")
                continue

            seizure_secs = get_seizure_onset(tsv_path)
            if seizure_secs is None or seizure_secs < 60:
                print(f"❌ Invalid seizure_secs: {seizure_secs} for {tsv_path}")
                continue

            df = convert_df_from_tsv_emg(edf_path, seizure_secs)
            if df is not None:
                print(f"📊 Label dist for {edf_path}: {df['label'].value_counts().sort_index().to_dict()}")

            if df is None or len(df) == 0:
                print("⚠️ No segments extracted")
                continue

            print(f"✅ Extracted {len(df)} segments from {fname}")
            all_segments.append(df)

if not all_segments:
    raise RuntimeError("❌ No usable EMG segments found.")

df_all = pd.concat(all_segments, ignore_index=True)

max_label_5 = 225000
label_5_df = df_all[df_all['label'] == 5]
other_labels_df = df_all[df_all['label'] != 5]

if len(label_5_df) > max_label_5:
    label_5_df = label_5_df.sample(n=max_label_5, random_state=42)

df_all = pd.concat([label_5_df, other_labels_df], ignore_index=True)

print(df_all['label'].value_counts(normalize=True))

print("Final label distribution:", df_all['label'].value_counts().sort_index())
print("Unique labels:", sorted(df_all['label'].unique()))

df_all["features"] = flatten_and_pool(df_all["features"].tolist(), target_len=50)

# --- Train/test split ---
X = df_all["features"].tolist()
y = df_all["label"].tolist()

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42, shuffle=True
)

# --- Train XGBoost ---
print(f"\n🚀 Training CatBoost on {len(X_train)} samples...")

clf = CatBoostClassifier(
    iterations=30,
    depth=6,
    learning_rate=0.2,
    loss_function='MultiClass',
    classes_count=6,
    eval_metric='Accuracy',
    verbose=False
)

n_epochs = 30
best_acc = 0

for epoch in range(n_epochs):
    clf.fit(X_train, y_train, init_model=clf if epoch > 0 else None)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📆 Epoch {epoch+1}/{n_epochs} | 🎯 Test Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc

    print(y_pred)
    print(y_test)
    
    joblib.dump(clf, f"emg_catboost_best_model_epoch{epoch}.joblib")

print(classification_report(y_test, y_pred, digits=3))

print(f"🏆 Best Accuracy: {best_acc:.4f}")