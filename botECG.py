import os
import numpy as np
import torch
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import PretrainedConfig, PreTrainedModel, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset
from preprocessEDF import convert_df_from_tsv
from TSVreader import find_matching_tsv, get_seizure_onset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import tensorflow as tf
from torch.utils.data import Dataset


def flatten_and_pool(arr_list, target_len=50):
    pooled = []
    for arr in arr_list:
        x = torch.tensor(arr).unsqueeze(0)  # shape: (1, channels, time)
        x = F.adaptive_avg_pool1d(x, target_len)  # shape: (1, channels, target_len)
        x = x.squeeze(0).flatten()  # shape: (channels * target_len,)
        pooled.append(x.numpy())
    return pooled

base_dir = "ds005873"
all_segments = []

subjects = [d for d in os.listdir(base_dir) if d.startswith("sub-") and os.path.isdir(os.path.join(base_dir, d))]
for sub in tqdm(subjects, desc="🔍 Subjects"):
    sub_path = os.path.join(base_dir, sub)

    sessions = [s for s in os.listdir(sub_path) if s.startswith("ses-")]
    for ses in tqdm(sessions, desc=f"  📁 {sub}", leave=False):
        ses_path = os.path.join(sub_path, ses)

        ecg_path = os.path.join(ses_path, "ecg")
        if not os.path.isdir(ecg_path):
            continue

        edf_files = [f for f in os.listdir(ecg_path) if f.endswith(".edf")]
        for fname in tqdm(edf_files, desc=f"    📄 {sub}/{ses}", leave=False):
            edf_path = os.path.join(ecg_path, fname)
            tsv_path = find_matching_tsv(edf_path)

            if not tsv_path:
                continue

            seizure_secs = get_seizure_onset(tsv_path)
            if seizure_secs is None:
                continue

            df = convert_df_from_tsv(edf_path, seizure_secs)
            if df is not None:
                all_segments.append(df)

if not all_segments:
    raise RuntimeError("❌ No usable ECG segments found.")

df_all = pd.concat(all_segments, ignore_index=True)

# Encode labels
le = LabelEncoder()
df_all["label"] = le.fit_transform(df_all["label"])

print("✅ Sample shape:", df_all['features'].iloc[0].shape)

# Split dataset
features = df_all["features"].tolist()
labels = df_all["label"].tolist()

pooled_features = flatten_and_pool(features, target_len=50)

X_train, X_test, y_train, y_test = train_test_split(
    pooled_features,
    labels,
    stratify=labels,
    test_size=0.2,
    random_state=1729
)

df_train = pd.DataFrame({"features": X_train, "label": y_train})
df_test = pd.DataFrame({"features": X_test, "label": y_test})

class ECGConfig(PretrainedConfig):
    def __init__(self, input_dim=50, num_labels=5, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_labels = num_labels

class ECGClassifier(PreTrainedModel):
    config_class = ECGConfig

    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_labels)
        )

    def forward(self, input_values=None, labels=None):
        logits = self.classifier(input_values)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
config = ECGConfig(input_dim=50, num_labels=5)
model = ECGClassifier(config)

model.save_pretrained("ecg_hf_model")
config.save_pretrained("ecg_hf_model")

training_args = TrainingArguments(
    output_dir="./ecg_output",
    per_device_train_batch_size=2,
    num_train_epochs=13,
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

def preprocess(example):
    x = np.array(example["features"]).astype(np.float32)
    return {
        "input_values": torch.tensor(x, dtype=torch.float32),
        "labels": torch.tensor(example["label"], dtype=torch.long)
    }

print(f"\n🚀 Starting training on {len(df_train)} samples...")

class ECGTorchDataset(Dataset):
    def __init__(self, df):
        self._features = [torch.tensor(f, dtype=torch.float32) for f in df["features"].tolist()]
        self._labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {
            "input_values": self._features[idx],
            "labels": self._labels[idx]
        }

dataset_train = ECGTorchDataset(df_train)
dataset_test = ECGTorchDataset(df_test)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics
)

trainer.train()

joblib.dump({
    "model": model,
    "label_encoder": le,
    "test_df": df_test
}, "FINAL_ecg_preds.joblib")

print("✅ Model trained and saved as ecg_preds.joblib")