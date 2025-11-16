import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import trange
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib
from preprocessEDF import convert_df, parse_seizure_info

def resample_to_fixed(data, target_timepoints=2560):
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        return None
    n_channels, original_len = data.shape
    if original_len == target_timepoints:
        return data
    return np.array([
        np.interp(np.linspace(0, original_len - 1, target_timepoints), np.arange(original_len), ch)
        for ch in data
    ])

SEIZURE_TYPES = {
    "IAS":  ["PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09", "PN11", "PN12", "PN13", "PN16", "PN17"],
    "FBTC": ["PN10"],
    "WIAS": ["PN14"]
}

def process_all_edfs(edf_files):
    features, labels, patient_ids = [], [], []

    for edf_path in edf_files:
        patient_dir = os.path.basename(os.path.dirname(edf_path))
        seizure_txt_path = os.path.join(os.path.dirname(edf_path), f"Seizures-list-{patient_dir}.txt")
        if not os.path.exists(seizure_txt_path):
            continue
        entries = parse_seizure_info(seizure_txt_path)
        fname = os.path.basename(edf_path).strip().lower()
        matching = [e for e in entries if e["edf_file"] == fname]
        if not matching:
            continue
        df, _ = convert_df(edf_path, matching)
        if df is None or df.empty:
            continue
        df["features"] = df["features"].apply(lambda x: resample_to_fixed(x, target_timepoints=2560))
        df = df[df["features"].notnull()]
        if df.empty:
            continue
        if isinstance(df["label"].iloc[0], np.ndarray):
            df["label"] = df["label"].apply(lambda x: np.argmax(x))

        for _, row in df.iterrows():
            data = row["features"].astype(np.float32)
            mean = data.mean(axis=1)
            std = data.std(axis=1)
            p25 = np.percentile(data, q=25, axis=1)
            p75 = np.percentile(data, q=75, axis=1)
            pooled = np.concatenate([
                np.pad(mean, (0, max(0, 49 - len(mean))), 'constant')[:49],
                np.pad(std,  (0, max(0, 49 - len(std))),  'constant')[:49],
                np.pad(p25,  (0, max(0, 49 - len(p25))),  'constant')[:49],
                np.pad(p75,  (0, max(0, 49 - len(p75))),  'constant')[:49]
            ])
            features.append(pooled.flatten())
            labels.append(row["label"])
            patient_ids.append(patient_dir)

    return np.array(features), np.array(labels), np.array(patient_ids)

for seizure_type, patient_list in SEIZURE_TYPES.items():
    print(f"\nStarting training for seizure type: {seizure_type}")

    base_dir = "siena-scalp-eeg-database-1.0.0"
    edf_files = [
        os.path.join(dp, f)
        for patient in patient_list
        for dp, _, files in os.walk(os.path.join(base_dir, patient))
        for f in files if f.endswith(".edf")
    ]

    X, y, patient_ids = process_all_edfs(edf_files)

    le = LabelEncoder()
    y = le.fit_transform(y)

    if len(patient_list) > 1:
        print("Patient-level split")
        train_patients, test_patients = train_test_split(patient_list, test_size=0.2, random_state=42)
        train_idx = np.isin(patient_ids, train_patients)
        test_idx = np.isin(patient_ids, test_patients)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    else:
        print("Only one patient")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rus = RandomUnderSampler(random_state=42)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
    X_test_bal, y_test_bal = rus.fit_resample(X_test, y_test)

    print(f"{seizure_type} Balanced train dist:", np.unique(y_train_bal, return_counts=True))
    print(f"{seizure_type} Balanced test dist:", np.unique(y_test_bal, return_counts=True))

    counts = Counter(y_train_bal)
    total = sum(counts.values())
    class_weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    boost_factors = {1: 1.5, 3: 1.5, 5: 0.5}
    for cls in class_weights:
        class_weights[cls] *= boost_factors.get(cls, 1.0)
    sample_weights = np.array([class_weights[y] for y in y_train_bal])

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": len(le.classes_),
        "max_depth": 3,
        "min_child_weight": 2,
        "gamma": 0.1,
        "eta": 0.05,
        "nthread": 4,
        "subsample": 0.6,
        "colsample_bytrees": 0.6,
    }

    n_epochs = 30
    acc_per_epoch = []
    model = None

    dtrain = xgb.DMatrix(X_train_bal, label=y_train_bal, weight=sample_weights)
    dtest = xgb.DMatrix(X_test_bal, label=y_test_bal)

    print("Training\n")
    for epoch in trange(1, n_epochs + 1, desc=f"Epochs [{seizure_type}]"):
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=10,
            xgb_model=model
        )

        y_proba = model.predict(dtest)
        y_logit = np.log(np.clip(y_proba, 1e-8, 1))
        y_pred = np.argmax(y_logit, axis=1)

        acc = accuracy_score(y_test_bal, y_pred)
        acc_per_epoch.append(acc)

        joblib.dump(model, f"{seizure_type.lower()}_xgb_epoch{epoch}.joblib")

        print(f"\n{seizure_type} | Epoch {epoch} | Accuracy: {acc:.4f}")
        print(classification_report(y_test_bal, y_pred, target_names=[f"Class {c}" for c in le.classes_]))

    print(f"\n{seizure_type} training complete.")
    print(f"{seizure_type} Best Accuracy: {max(acc_per_epoch):.4f}")