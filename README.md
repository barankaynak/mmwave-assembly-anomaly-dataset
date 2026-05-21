# Dataset: A Non-Invasive Monitoring Framework for Computer-Integrated Manual Assembly

[![DOI](https://zenodo.org/badge/1245400645.svg)](https://doi.org/10.5281/zenodo.20323005)

**Paper:** "A Non-Invasive Monitoring Framework for Computer-Integrated Manual Assembly: mmWave Sensing with Deep Temporal Models"  
**Authors:** Baran Kaynak, Sümeyye Kaynak, Andrew Kusiak  
**Journal:** International Journal of Computer Integrated Manufacturing (submitted 2026)  
**Funding:** TUBITAK 2219 Research Program, Project No. 1059B1923025

---

## Repository Structure

```
dataset/
├── sensor_config/
│   └── iwr6843isk_radar_profile.cfg      <- TI mmWave Studio radar config (all sessions)
├── data/
│   ├── session_01_training/
│   │   ├── mmwave_data.csv               <- Raw point-cloud frames from IWR6843ISK
│   │   ├── distance_sensors.csv          <- Raw ToF distance readings (boxes 1-5)
│   │   ├── merged_df.csv                 <- Synchronized merge with ground-truth labels
│   │   ├── video_frames.csv              <- Video frame timestamps (sync reference)
│   │   └── experiment_info.txt           <- Session metadata (date, SW versions)
│   ├── session_02_normal_seq125/         <- (same 5 files per session)
│   ├── session_03_normal_seq135/
│   ├── session_04_normal_seq145/
│   ├── session_05_anomaly_fast/
│   └── session_06_anomaly_slow/
├── models/
│   └── best_cnnlstm_model_vel_v2.keras   <- Trained CNN-GRU state estimator
├── notebooks/
│   ├── 00_dataprepare_all.ipynb          <- Data loading and preprocessing
│   ├── 01_mmwave_cnn_lstm_model.ipynb    <- CNN-GRU training and evaluation
│   ├── 03_tft_anomaly_v2.ipynb           <- TFT + Z-score anomaly detection
│   └── 04_ablation_and_benchmark.ipynb   <- Ablation study and inference benchmark
├── scripts/
│   └── run_all_pt.py                     <- PyTorch ablation + benchmark runner
└── results/
    ├── ablation_results.json             <- CNN-GRU ablation (6 variants)
    ├── benchmark_results.json            <- Inference latency (CPU + GPU)
    └── threshold_sweep.json              <- Threshold sensitivity analysis
```

---

## Experimental Sessions

All data was collected on **2025-05-15** using:
- **mmWave radar:** Texas Instruments IWR6843ISK (60 GHz, on-chip point-cloud output, no DCA1000EVM)
- **Distance sensors:** VL53L0X-V2 Time-of-Flight sensors (×3, training-time labelling oracle only)
- **Task:** Repetitive manual assembly — operator picks boxes from 5 labelled stations in a defined sequence

| Session folder | Scenario | Role |
|---|---|---|
| `session_01_training` | All box types, normal pacing | Training data for CNN-GRU and TFT |
| `session_02_normal_seq125` | Correct sequence (1→2→5) | Normal evaluation |
| `session_03_normal_seq135` | Wrong sequence (1→3→5) | Sequence anomaly |
| `session_04_normal_seq145` | Wrong sequence (1→4→5) | Sequence anomaly |
| `session_05_anomaly_fast` | Correct sequence, rushed | Pacing anomaly (fast) |
| `session_06_anomaly_slow` | Correct sequence, prolonged | Pacing anomaly (slow) |

---

## File Descriptions

### `sensor_config/iwr6843isk_radar_profile.cfg`
TI mmWave Studio configuration file used for all sessions. Key parameters: 60 GHz, range resolution 0.044 m, max range 3.95 m, max radial velocity ±1 m/s, frame duration 100 ms (~10 Hz). Load directly into mmWave Studio or the TI SDK to replicate the hardware setup.

### Per-session files (5 files per session)

| File | Description |
|---|---|
| `mmwave_data.csv` | **Raw** point-cloud frames from IWR6843ISK. Columns: `frame_id`, `timestamp`, `x`, `y`, `z`, `snr`, `velocity`, `range`. One row per detected point per frame. |
| `distance_sensors.csv` | **Raw** ToF distance readings. Columns: `frame_id`, `timestamp`, `box1_distance` … `box5_distance` (mm). Training-time labelling oracle — not used at runtime. |
| `video_frames.csv` | Video frame timestamps for temporal alignment. Columns: `frame_id`, `timestamp`, `video_frame_number`. No video imagery is stored. |
| `merged_df.csv` | Synchronized merge of mmWave and ToF data at the frame level, with derived `box_state` ground-truth labels. Primary input to all notebooks. |
| `experiment_info.txt` | Session metadata: recording date/time, Python version, and full package dependency list for the data-acquisition software. |

---

## How to Reproduce Results

### Requirements
```
python >= 3.10
tensorflow >= 2.12       # for CNN-GRU (.keras model)
torch >= 2.0             # for PyTorch ablation script
pytorch-forecasting      # for TFT
pandas, numpy, scikit-learn, matplotlib
```

### Steps
1. **Data preparation:** Run `notebooks/00_dataprepare_all.ipynb`
2. **CNN-GRU training:** Run `notebooks/01_mmwave_cnn_lstm_model.ipynb` (or load pre-trained weights from `models/best_cnnlstm_model_vel_v2.keras`)
3. **Anomaly detection:** Run `notebooks/03_tft_anomaly_v2.ipynb`
4. **Ablation + benchmark:** Run `python scripts/run_all_pt.py` (PyTorch, no GPU required for ablation)

---

## Key Results (from paper)

| Metric | Value |
|---|---|
| CNN-GRU state classification accuracy | 93.66% (6 classes) |
| Sequence anomaly detection (Seq 145) | 75.14% |
| Pacing anomaly detection (Slow) | 86.44% (either-channel) |
| Inference latency (CPU, Ryzen 7 5800X) | 12.6 ms/frame median |
| Inference latency (GPU, RTX 3090) | 0.51 ms/frame median |

---

## License

The dataset is released under **CC BY 4.0**. The code is released under **MIT License**.

---

## Repository

**GitHub:** https://github.com/barankaynak/mmwave-assembly-anomaly-dataset  
**Zenodo DOI:** https://doi.org/10.5281/zenodo.20323005

---

## Citation

If you use this dataset or code, please cite the paper **and** the dataset record:

### Cite the paper

```bibtex
@article{kaynak2026mmwave,
  author  = {Kaynak, Baran and Kaynak, S\"{u}meyye and Kusiak, Andrew},
  title   = {A Non-Invasive Monitoring Framework for Computer-Integrated Manual
             Assembly: {mmWave} Sensing with Deep Temporal Models},
  journal = {International Journal of Computer Integrated Manufacturing},
  year    = {2026},
  note    = {submitted}
}
```

### Cite the dataset

```bibtex
@dataset{kaynak2026dataset,
  author    = {Kaynak, Baran and Kaynak, S\"{u}meyye and Kusiak, Andrew},
  title     = {{mmWave} Assembly Anomaly Detection Dataset},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.20323005},
  url       = {https://doi.org/10.5281/zenodo.20323005}
}
```

### Plain-text (APA)

Kaynak, B., Kaynak, S., & Kusiak, A. (2026). *mmWave Assembly Anomaly Detection Dataset* (v1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.20323005
