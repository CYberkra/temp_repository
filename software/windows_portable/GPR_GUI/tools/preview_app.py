#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch GPR GUI with sample data for screenshot (headless-safe)."""
import os
import time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
import sys
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
REPO_GPR = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "repos", "GPR_GUI"))
if REPO_GPR not in sys.path:
    sys.path.append(REPO_GPR)

from app import GPRGui, detect_csv_header

SAMPLE_PATH = os.path.join(ROOT_DIR, "sample_data", "sample_bscan.csv")


def load_sample(path):
    header_info = detect_csv_header(path)
    skip_lines = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(10):
            line = f.readline()
            if not line:
                break
            if "=" in line or "Samples" in line or "Traces" in line:
                skip_lines = i + 1
    df = pd.read_csv(path, header=None, skiprows=skip_lines)
    raw_data = df.values
    if header_info:
        samples = header_info["a_scan_length"]
        traces = header_info["num_traces"]
        if raw_data.shape[1] <= 10 and raw_data.shape[0] >= samples * traces:
            col_idx = 3 if raw_data.shape[1] > 3 else raw_data.shape[1] - 1
            signal_1d = raw_data[:, col_idx]
            data = signal_1d[:traces * samples].reshape((traces, samples)).T
        elif raw_data.shape[0] == traces and raw_data.shape[1] >= samples:
            data = raw_data[:, :samples].T
        elif raw_data.shape[0] >= samples and raw_data.shape[1] >= traces:
            data = raw_data[:samples, :traces]
        else:
            data = raw_data
    else:
        data = raw_data
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=np.nanmean(data))
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data, header_info


def main():
    app = GPRGui()
    data, header_info = load_sample(SAMPLE_PATH)
    app.data = data
    app.data_path = SAMPLE_PATH
    app.header_info = header_info
    app.original_data = data.copy()
    app.history = []

    app.cmap_var.set("seismic")
    app.cmap_invert_var.set(False)
    app.show_cbar_var.set(True)
    app.show_grid_var.set(True)
    app.percentile_var.set(True)
    app.p_low_var.set("2")
    app.p_high_var.set("98")
    app.display_downsample_var.set(True)
    app.display_max_samples_var.set("500")
    app.display_max_traces_var.set("200")

    app.plot_data(data)
    app.update_idletasks()
    app.update()

    # keep window alive for external screenshot capture
    time.sleep(6)
    app.destroy()


if __name__ == "__main__":
    main()
