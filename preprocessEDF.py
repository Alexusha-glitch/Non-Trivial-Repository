import numpy as np
import pandas as pd
import re
import os
from datetime import datetime, timedelta
import mne

fs = 512
mne.set_log_level('WARNING')

def parse_seizure_info(path):
    with open(path, 'r') as file:
        content = file.read()

    seizure_entries = []
    blocks = re.split(r"\n\s*\n", content)

    for block in blocks:
        file_match = re.search(r'File name: (\S+)', block)
        start_match = re.search(r'Registration start time: *([^\n\r]+)', block)
        seizure_match = re.search(r'Seizure start time: *([^\n\r]+)', block)
        sleep_flag = "(in sleep)" in block.lower()

        if file_match and start_match and seizure_match:
            edf_file = file_match.group(1).strip().lower()
            reg_start_str = start_match.group(1).strip()
            seizure_str = seizure_match.group(1).strip()

            for fmt in ['%H.%M.%S', '%H:%M:%S', '%H:%M.%S', '%H.%M:%S']:
                try:
                    reg_start = datetime.strptime(reg_start_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                print(f"⚠️ Unrecognized registration start time: '{reg_start_str}'")
                continue

            for fmt in ['%H.%M.%S', '%H:%M:%S', '%H:%M.%S', '%H.%M:%S']:
                try:
                    seizure_time = datetime.strptime(seizure_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                print(f"⚠️ Unrecognized seizure start time: '{seizure_str}'")
                continue

            seizure_entries.append({
                'edf_file': edf_file,
                'start_time': reg_start,
                'seizure_time': seizure_time,
                'in_sleep': sleep_flag
            })

    return seizure_entries


preictal_bins = [
    (0, 600),
    (600, 1200),
    (1200, 1800),
    (1800, 2400),
    (2400, 3600),
    (3600, 999999999999999999)
]

def convert_df(edf_path, seizure_entries):
    print(f"\nReading EDF file: {edf_path}")

    # Find matching seizure entry
    fname = os.path.basename(edf_path).strip().lower()
    entry = next((e for e in seizure_entries if e['edf_file'].strip().lower() == fname), None)
    if not entry:
        print(f"No matching seizure annotation for {fname}")
        return None, None

    seizure_time = entry['seizure_time']
    start_time = entry['start_time']
    delta_secs = int((seizure_time - start_time).total_seconds())

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
        raw.pick_types(eeg=True)
        raw.resample(fs)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return None, None

    print(f"Preictal duration: {delta_secs} seconds")

    all_segments, all_labels = [], []
    segment_len = fs * 5  # 2 seconds

    for label, (start_sec, end_sec) in enumerate(preictal_bins):
        if delta_secs < start_sec:
            continue  # Not enough data before seizure

        # Time range in seconds (absolute from start of recording)
        effective_end = min(end_sec, delta_secs)

        abs_start = delta_secs - effective_end
        abs_end = delta_secs - start_sec

        start_sample = int(abs_start * fs)
        end_sample = int(abs_end * fs)

        # Read only that segment from disk
        try:
            raw_chunk = raw.copy().crop(tmin=abs_start, tmax=abs_end, include_tmax=False)
            signal = raw_chunk.get_data()
        except Exception as e:
            print(f"Failed to extract chunk: {e}")
            continue

        if signal.shape[1] < segment_len:
            continue

        for i in range(0, signal.shape[1] - segment_len + 1, segment_len):
            segment = signal[:, i:i+segment_len]
            all_segments.append(segment)
            all_labels.append(label)

    if not all_segments:
        print("No valid segments")
        return None, None

    df = pd.DataFrame({
        "features": all_segments,
        "label": all_labels
    })
    return df, entry['in_sleep']

def convert_df_from_tsv(edf_path, seizure_secs):
    fs_ecg = 250
    print(f"\nReading EDF file: {edf_path}")

    # Use 00:00:00 as assumed recording start
    start_time = datetime.strptime("00.00.00", "%H.%M.%S")
    seizure_time = start_time + timedelta(seconds=seizure_secs)
    delta_secs = int((seizure_time - start_time).total_seconds())

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
        raw.pick_channels([ch for ch in raw.ch_names if "emg" in ch.lower()])
        raw.resample(fs_ecg)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return None

    print(f"📏 Preictal duration: {delta_secs} seconds")

    all_segments, all_labels = [], []
    segment_len = fs_ecg * 5  # 2 seconds

    for label, (start_sec, end_sec) in enumerate(preictal_bins):
        if delta_secs < end_sec:
            continue

        abs_start = delta_secs - end_sec
        abs_end = delta_secs - start_sec

        try:
            raw_chunk = raw.copy().crop(tmin=abs_start, tmax=abs_end, include_tmax=False)
            signal = raw_chunk.get_data()
        except Exception as e:
            print(f"Failed to extract chunk: {e}")
            continue

        if signal.shape[1] < segment_len:
            continue

        for i in range(0, signal.shape[1] - segment_len + 1, segment_len):
            segment = signal[:, i:i+segment_len]
            all_segments.append(segment)
            all_labels.append(label)

    if not all_segments:
        print("No valid segments")
        return None

    return pd.DataFrame({
        "features": all_segments,
        "label": all_labels
    })

def convert_df_from_tsv_emg(edf_path, seizure_secs):
    fs = 25
    segment_duration = 5
    segment_len = fs * segment_duration

    print(f"\nReading EDF file: {edf_path}")

    # Use 00:00:00 as assumed recording start
    start_time = datetime.strptime("00.00.00", "%H.%M.%S")
    seizure_time = start_time + timedelta(seconds=seizure_secs)
    delta_secs = int((seizure_time - start_time).total_seconds())

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
        print(f"Raw channels: {raw.ch_names}")
        
        # Pick EMG channels only
        emg_channels = [ch for ch in raw.ch_names if "emg" in ch.lower()]
        if not emg_channels:
            print("No EMG channels found.")
            return None

        raw.pick_channels(emg_channels)
        print(f"Picked EMG channels: {raw.ch_names}")
        raw.resample(fs)
    except Exception as e:
        print(f"Failed to load or resample file: {e}")
        return None

    print(f"Preictal duration: {delta_secs} seconds")

    all_segments, all_labels = [], []

    for label, (start_sec, end_sec) in enumerate(preictal_bins):
        if delta_secs < start_sec:
            continue  # Not enough data for this bin

        effective_end = min(end_sec, delta_secs)

        abs_start = delta_secs - effective_end
        abs_end = delta_secs - start_sec

        try:
            raw_chunk = raw.copy().crop(tmin=abs_start, tmax=abs_end, include_tmax=False)
            signal = raw_chunk.get_data()
        except Exception as e:
            print(f"Failed to extract chunk: {e}")
            continue

        print(f"Extracted window: {signal.shape}")

        if signal.shape[1] < segment_len:
            print("Skipping bin — not enough data for even 1 segment.")
            continue

        for i in range(0, signal.shape[1] - segment_len + 1, segment_len):
            segment = signal[:, i:i+segment_len]
            all_segments.append(segment)
            all_labels.append(label)

    if not all_segments:
        print("No valid EMG segments found in bins")
        return None

    print(f"Final EMG segments: {len(all_segments)}")
    return pd.DataFrame({
        "features": all_segments,
        "label": all_labels
    })