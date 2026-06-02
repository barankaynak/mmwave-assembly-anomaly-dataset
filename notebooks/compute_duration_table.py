#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_duration_table.py
-------------------------
Reproduces the State Duration Anomaly Detection (Z-score) table for ALL FIVE
scenarios, using the EXACT logic extracted from 03_tft_anomaly_v2.ipynb
(PART 1 CNN-GRU inference + PART 3 duration Z-score). The TFT (PART 2) is
intentionally skipped because the duration channel does not depend on it.

Goal: obtain the missing Seq 135 / Seq 145 rows for Table tab:dur_anomaly.
Validation: the Normal / Fast / Slow rows must match the published table
  Normal: count 221, meanZ 0.6584, stdZ 0.7502, maxZ 4.5667, %|Z|>1 21.27
  Fast  : count 198, meanZ 0.9179, stdZ 0.5695, maxZ 2.5888, %|Z|>1 38.89
  Slow  : count 315, meanZ 1.3474, stdZ 1.0857, maxZ 6.8661, %|Z|>1 58.10
If those three match, the Seq135/Seq145 rows are trustworthy.

Run (from the dataset/notebooks/ directory, on a machine with tensorflow):
    python compute_duration_table.py
Only requires: tensorflow, numpy, pandas  (NO torch / pytorch-forecasting).
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Paths resolved relative to THIS file (so CWD does not matter)
# ---------------------------------------------------------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # running pasted into a Jupyter cell (no __file__): assume CWD is dataset/notebooks/
    SCRIPT_DIR = os.getcwd()
# If the auto-resolved data dir does not exist, override these two lines manually:
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CNNGRU_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "best_cnnlstm_model_vel_v2.keras")

NORMAL_DATA_FILE_PATH_COMMON = os.path.join(BASE_DATA_DIR, "session_02_normal_seq125", "merged_df.csv")
ANOMALY_DATA_FILE_PATHS = [
    os.path.join(BASE_DATA_DIR, "session_03_normal_seq135", "merged_df.csv"),
    os.path.join(BASE_DATA_DIR, "session_04_normal_seq145", "merged_df.csv"),
    os.path.join(BASE_DATA_DIR, "session_05_anomaly_fast", "merged_df.csv"),
    os.path.join(BASE_DATA_DIR, "session_06_anomaly_slow", "merged_df.csv"),
]
ANOMALY_DATA_LABELS = ["Anomaly Seq 135", "Anomaly Seq 145", "Anomaly Fast", "Anomaly Slow"]

# EXACT config from the notebook
CNN_GRU_INPUT_CONFIG = {
    "FRAME_ID_COLUMN": "timestamp_mmwave", "IMG_HEIGHT": 48, "IMG_WIDTH": 48, "NUM_CHANNELS": 2,
    "SEQUENCE_LENGTH": 5, "X_MIN_METERS": -0.5, "X_MAX_METERS": 0.5,
    "Y_MIN_METERS": 0.1, "Y_MAX_METERS": 1.0, "X_BINS": 48, "Y_BINS": 48,
}
DURATION_Z_THRESHOLD = 1.0

# ===========================================================================
# PART 1 functions (verbatim from 03_tft_anomaly_v2.ipynb)
# ===========================================================================
def load_cnngru_model(model_path):
    print(f"Loading CNN-GRU model from: {os.path.abspath(model_path)}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN-GRU model file not found: {os.path.abspath(model_path)}")
    model = load_model(model_path, compile=False)
    print("CNN-GRU model loaded successfully.")
    return model

def create_heatmaps_from_points(df_points, config):
    all_single_frame_heatmaps = []
    unique_frame_ids_sorted = sorted(df_points[config['FRAME_ID_COLUMN']].unique())
    grouped_points = df_points.groupby(config['FRAME_ID_COLUMN'])
    for frame_id_val in unique_frame_ids_sorted:
        if frame_id_val not in grouped_points.groups:
            heatmap = np.zeros((config['Y_BINS'], config['X_BINS'], config['NUM_CHANNELS']), dtype=np.float32)
            all_single_frame_heatmaps.append(heatmap)
            continue
        points_in_frame_df = grouped_points.get_group(frame_id_val)
        current_heatmap = np.zeros((config['Y_BINS'], config['X_BINS'], config['NUM_CHANNELS']), dtype=np.float32)
        if 'x' not in points_in_frame_df.columns or 'y' not in points_in_frame_df.columns:
            all_single_frame_heatmaps.append(current_heatmap); continue
        x_coords, y_coords = points_in_frame_df['x'].to_numpy(), points_in_frame_df['y'].to_numpy()
        x_indices = np.floor((x_coords - config['X_MIN_METERS']) / (config['X_MAX_METERS'] - config['X_MIN_METERS']) * config['X_BINS']).astype(int)
        y_indices = np.floor((y_coords - config['Y_MIN_METERS']) / (config['Y_MAX_METERS'] - config['Y_MIN_METERS']) * config['Y_BINS']).astype(int)
        valid_indices = (x_indices >= 0) & (x_indices < config['X_BINS']) & (y_indices >= 0) & (y_indices < config['Y_BINS'])
        x_indices_valid, y_indices_valid = x_indices[valid_indices], y_indices[valid_indices]
        points_in_frame_valid = points_in_frame_df[valid_indices]
        if not points_in_frame_valid.empty:
            snr_values = points_in_frame_valid.get('snr', pd.Series(1.0, index=points_in_frame_valid.index)).fillna(1.0).to_numpy()
            np.maximum.at(current_heatmap[:, :, 0], (y_indices_valid, x_indices_valid), snr_values)
            if config['NUM_CHANNELS'] > 1:
                velocity_values = points_in_frame_valid.get('doppler', pd.Series(0.0, index=points_in_frame_valid.index)).fillna(0.0).to_numpy()
                temp_vel_sum = np.zeros((config['Y_BINS'], config['X_BINS']), dtype=np.float32)
                temp_vel_count = np.zeros((config['Y_BINS'], config['X_BINS']), dtype=np.int32)
                np.add.at(temp_vel_sum, (y_indices_valid, x_indices_valid), velocity_values)
                np.add.at(temp_vel_count, (y_indices_valid, x_indices_valid), 1)
                current_heatmap[:, :, 1] = np.divide(temp_vel_sum, temp_vel_count, out=np.zeros_like(temp_vel_sum), where=temp_vel_count != 0)
        all_single_frame_heatmaps.append(current_heatmap)
    if not all_single_frame_heatmaps:
        return np.array([]), {}
    return np.array(all_single_frame_heatmaps), {fid: i for i, fid in enumerate(unique_frame_ids_sorted)}

def create_sequences_from_heatmaps(heatmaps, sequence_length):
    if len(heatmaps) < sequence_length:
        return np.array([])
    return np.array([heatmaps[i: i + sequence_length] for i in range(len(heatmaps) - sequence_length + 1)])

def predict_box_interactions_cnngru(df_mmwave_points_input, cnngru_model, cnn_gru_config):
    print("Starting box prediction using CNN-GRU...")
    df_mmwave_points = df_mmwave_points_input.copy()
    if df_mmwave_points.empty:
        return pd.DataFrame(columns=['timestamp_numeric', 'predicted_box'])
    heatmaps_all_frames, frame_id_to_idx_map = create_heatmaps_from_points(df_mmwave_points, cnn_gru_config)
    if heatmaps_all_frames.size == 0:
        return pd.DataFrame(columns=['timestamp_numeric', 'predicted_box'])
    sequences = create_sequences_from_heatmaps(heatmaps_all_frames, cnn_gru_config['SEQUENCE_LENGTH'])
    if sequences.size == 0:
        return pd.DataFrame(columns=['timestamp_numeric', 'predicted_box'])
    sequences = sequences.astype(np.float32)
    predictions_proba = cnngru_model.predict(sequences, verbose=0)
    predicted_classes = np.argmax(predictions_proba, axis=1)
    print(f"  CNN-GRU prediction complete. Found classes: {np.unique(predicted_classes)}")
    idx_to_frame_id_map = {v: k for k, v in frame_id_to_idx_map.items()}
    used_frame_indices = sorted(idx_to_frame_id_map.keys())
    unique_frames_used_for_heatmaps = [idx_to_frame_id_map[i] for i in used_frame_indices]
    unique_frames_timestamps_df = df_mmwave_points[df_mmwave_points[cnn_gru_config['FRAME_ID_COLUMN']].isin(unique_frames_used_for_heatmaps)]
    unique_frames_timestamps_df = unique_frames_timestamps_df[[cnn_gru_config['FRAME_ID_COLUMN'], 'timestamp_numeric']].drop_duplicates(subset=[cnn_gru_config['FRAME_ID_COLUMN']]).sort_values(cnn_gru_config['FRAME_ID_COLUMN']).reset_index(drop=True)
    results_list = []
    for i in range(len(sequences)):
        last_frame_in_sequence_heatmap_idx = i + cnn_gru_config['SEQUENCE_LENGTH'] - 1
        if last_frame_in_sequence_heatmap_idx < len(unique_frames_timestamps_df):
            timestamp_val = unique_frames_timestamps_df['timestamp_numeric'].iloc[last_frame_in_sequence_heatmap_idx]
            results_list.append({'timestamp_numeric': timestamp_val, 'predicted_box': predicted_classes[i]})
    return pd.DataFrame(results_list).sort_values('timestamp_numeric').reset_index(drop=True)

def extract_state_changes_and_features(df_predicted_boxes, group_id_val="assembly_line_1"):
    if df_predicted_boxes.empty:
        return pd.DataFrame()
    print(f"Extracting state changes from {len(df_predicted_boxes)} records for group '{group_id_val}'...")
    changed_records = []
    last_box, last_timestamp = -999, np.nan
    for _, row in df_predicted_boxes.iterrows():
        current_box, current_timestamp = row['predicted_box'], row['timestamp_numeric']
        if current_box != last_box:
            box_duration = current_timestamp - last_timestamp if last_box != -999 and not np.isnan(last_timestamp) else 0.0
            changed_records.append({
                'timestamp_numeric': current_timestamp,
                'predicted_box': current_box,
                'prev_box': last_box if last_box != -999 else 0,
                'prev_box_duration_s': box_duration
            })
            last_box, last_timestamp = current_box, current_timestamp
    df_state_changes = pd.DataFrame(changed_records)
    if df_state_changes.empty:
        return df_state_changes
    df_state_changes['time_idx'] = range(len(df_state_changes))
    df_state_changes['group_id'] = group_id_val
    for col in ['predicted_box', 'prev_box']:
        df_state_changes[col] = df_state_changes[col].astype(int)
    df_state_changes['prev_box_duration_s'] = df_state_changes['prev_box_duration_s'].astype(float)
    print(f"  Reduced to {len(df_state_changes)} state changes.")
    return df_state_changes.reset_index(drop=True)

def preprocess_raw_data(data_path, frame_id_col='timestamp_mmwave'):
    print(f"Loading raw data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERR: File not found: {data_path}"); return pd.DataFrame()
    original_fid_col = frame_id_col
    if frame_id_col not in df.columns:
        frame_id_col = 'timestamp' if 'timestamp' in df.columns else None
    if not frame_id_col:
        print(f"ERR: Frame ID col '{original_fid_col}' or 'timestamp' not found."); return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(df[frame_id_col]):
        df[frame_id_col] = pd.to_datetime(df[frame_id_col], errors='coerce')
    df = df.dropna(subset=[frame_id_col])
    if df.empty:
        print("DataFrame empty post datetime/dropna."); return pd.DataFrame()
    df = df.sort_values(frame_id_col).reset_index(drop=True)
    df['timestamp_numeric'] = (df[frame_id_col] - df[frame_id_col].iloc[0]).dt.total_seconds()
    if original_fid_col != frame_id_col and original_fid_col not in df.columns:
        df[original_fid_col] = df[frame_id_col]
    print(f"  Preprocessing complete for {data_path}. Shape: {df.shape}")
    return df

# ===========================================================================
# PART 3 functions (verbatim from the notebook)
# ===========================================================================
def calculate_normal_duration_stats(df_normal_tft_data, min_samples_for_stats=3, default_std_epsilon=1e-6):
    if df_normal_tft_data.empty or not all(col in df_normal_tft_data.columns for col in ['prev_box', 'prev_box_duration_s', 'time_idx', 'group_id']):
        print("Warning: Normal TFT data for duration stats is empty or missing required columns.")
        return pd.DataFrame(columns=['prev_box', 'mean_duration', 'std_duration', 'count'])
    df_for_stats = df_normal_tft_data[~((df_normal_tft_data['time_idx'] == 0) & (df_normal_tft_data['prev_box_duration_s'] == 0.0))].copy()
    if df_for_stats.empty:
        print("Warning: No valid records remaining for duration stats after filtering initial state records.")
        return pd.DataFrame(columns=['prev_box', 'mean_duration', 'std_duration', 'count'])
    duration_stats = df_for_stats.groupby('prev_box')['prev_box_duration_s'].agg(
        mean_duration='mean', std_duration='std', count='count'
    ).reset_index()
    duration_stats['std_duration'] = np.maximum(duration_stats['std_duration'].fillna(default_std_epsilon), default_std_epsilon)
    reliable_stats = duration_stats[duration_stats['count'] >= min_samples_for_stats].copy()
    print(f"Calculated reliable duration statistics for {len(reliable_stats)} 'prev_box' types.")
    return reliable_stats

def score_duration_anomalies(df_tft_data, normal_duration_stats, default_std_epsilon=1e-6):
    output_cols = ['duration_z_score', 'mean_duration', 'std_duration', 'count']
    if df_tft_data.empty:
        return df_tft_data.assign(**{col: np.nan for col in output_cols})
    if normal_duration_stats.empty:
        print("Warning: Normal duration stats are empty.")
        return df_tft_data.assign(**{col: np.nan for col in output_cols})
    original_index = df_tft_data.index
    df_scored = pd.merge(df_tft_data.reset_index(drop=True), normal_duration_stats, on='prev_box', how='left')
    safe_std_duration = np.maximum(df_scored['std_duration'].fillna(default_std_epsilon), default_std_epsilon)
    df_scored['duration_z_score'] = (df_scored['prev_box_duration_s'] - df_scored['mean_duration']) / safe_std_duration
    df_scored.index = original_index
    return df_scored

# ===========================================================================
# Main: build state changes for every scenario, score, summarise
# ===========================================================================
def build_state_changes(path, label, model):
    raw = preprocess_raw_data(path, CNN_GRU_INPUT_CONFIG['FRAME_ID_COLUMN'])
    if raw.empty:
        return pd.DataFrame()
    preds = predict_box_interactions_cnngru(raw, model, CNN_GRU_INPUT_CONFIG)
    if preds.empty:
        return pd.DataFrame()
    return extract_state_changes_and_features(preds, group_id_val=label)

def summarise(df_scored):
    valid_z = df_scored['duration_z_score'].dropna()
    n = len(valid_z)
    if n == 0:
        return n, float('nan'), float('nan'), float('nan'), float('nan')
    abs_z = valid_z.abs()
    pct = (abs_z > DURATION_Z_THRESHOLD).mean() * 100.0
    return n, abs_z.mean(), abs_z.std(), abs_z.max(), pct

def main():
    model = load_cnngru_model(CNNGRU_MODEL_PATH)

    # Normal (session_02) -> also the source of the per-box mean/std
    print("\n========== NORMAL (session_02) ==========")
    df_norm_tft = build_state_changes(NORMAL_DATA_FILE_PATH_COMMON, "normal_sequence", model)
    normal_stats = calculate_normal_duration_stats(df_norm_tft, min_samples_for_stats=3)

    datasets = [("Normal Data Eval", df_norm_tft)]
    for path, label in zip(ANOMALY_DATA_FILE_PATHS, ANOMALY_DATA_LABELS):
        print(f"\n========== {label} ==========")
        safe = label.replace(' ', '_').lower()
        datasets.append((label, build_state_changes(path, f"anomaly_{safe}", model)))

    # Score & summarise
    rows = []
    for label, df in datasets:
        if df.empty:
            rows.append((label, 0, float('nan'), float('nan'), float('nan'), float('nan')))
            continue
        scored = score_duration_anomalies(df.copy(), normal_stats)
        n, meanz, stdz, maxz, pct = summarise(scored)
        rows.append((label, n, meanz, stdz, maxz, pct))

    # ---- Print final table (paper columns) ----
    print("\n\n" + "=" * 92)
    print("STATE DURATION ANOMALY TABLE  (Z-score, |Z| > 1.0)")
    print("=" * 92)
    hdr = ("Dataset", "TotalStateChanges", "MeanAbsZ", "StdAbsZ", "MaxAbsZ", "%|Z|>1.0")
    print("%-20s %18s %10s %10s %10s %10s" % hdr)
    print("-" * 92)
    for label, n, meanz, stdz, maxz, pct in rows:
        print("%-20s %18d %10.4f %10.4f %10.4f %9.2f%%" % (label, n, meanz, stdz, maxz, pct))
    print("=" * 92)
    print("\nVALIDATION (must match the published table):")
    print("  Normal -> 221 / 0.6584 / 0.7502 / 4.5667 / 21.27%")
    print("  Fast   -> 198 / 0.9179 / 0.5695 / 2.5888 / 38.89%")
    print("  Slow   -> 315 / 1.3474 / 1.0857 / 6.8661 / 58.10%")
    print("If those 3 match, the Seq 135 / Seq 145 rows are trustworthy. Paste the whole table back.")

if __name__ == "__main__":
    main()
