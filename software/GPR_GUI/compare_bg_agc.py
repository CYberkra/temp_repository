#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline vs background suppression vs AGC improvement for A8-NEW-1.csv
Baseline: agcGain + subtracting_average_2D
BG improvement: SVD low-rank background removal (rank=1) on AGC output
AGC improvement: larger window + gain cap, then subtracting_average_2D
"""
import os
import sys
import numpy as np

# add module paths
BASE = os.path.dirname(os.path.abspath(__file__))
REPO_PYMOD = os.path.join(os.path.dirname(BASE), "PythonModule")
if REPO_PYMOD not in sys.path:
    sys.path.insert(0, REPO_PYMOD)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from read_file_data import readcsv, savecsv, save_image
from agcGain import agcGain
from subtracting_average_2D import subtracting_average_2D


def ensure_matrix(csv_path: str, matrix_path: str, nsamp: int = 801):
    if os.path.exists(matrix_path):
        return matrix_path
    # read raw csv: 5 columns, use 4th column (index 3)
    data = readcsv(csv_path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Unexpected CSV shape: {data.shape}")
    col = data[:, 3]
    # drop NaN rows
    col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    ntraces = len(col) // nsamp
    if ntraces * nsamp != len(col):
        col = col[: ntraces * nsamp]
    mat = col.reshape(nsamp, ntraces)
    savecsv(mat, matrix_path)
    return matrix_path


def svd_lowrank_remove(mat: np.ndarray, rank: int = 1):
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    r = max(1, int(rank))
    lowrank = (U[:, :r] * S[:r]) @ Vt[:r, :]
    return mat - lowrank


def agc_with_cap(mat: np.ndarray, window: int = 301, max_gain: float = 10.0):
    eps = 1e-8
    totsamps, tottraces = mat.shape
    if window > totsamps:
        energy = np.maximum(np.linalg.norm(mat, axis=0), eps)
        gain = np.minimum(1.0 / energy, max_gain)
        return mat * gain
    out = np.zeros_like(mat)
    halfwid = int(window / 2.0)
    # first samples
    energy = np.maximum(np.linalg.norm(mat[0:halfwid + 1, :], axis=0), eps)
    gain = np.minimum(1.0 / energy, max_gain)
    out[0:halfwid + 1, :] = mat[0:halfwid + 1, :] * gain
    for smp in range(halfwid, totsamps - halfwid + 1):
        winstart = int(smp - halfwid)
        winend = int(smp + halfwid)
        energy = np.maximum(np.linalg.norm(mat[winstart:winend + 1, :], axis=0), eps)
        gain = np.minimum(1.0 / energy, max_gain)
        out[smp, :] = mat[smp, :] * gain
    energy = np.maximum(np.linalg.norm(mat[totsamps - halfwid:totsamps + 1, :], axis=0), eps)
    gain = np.minimum(1.0 / energy, max_gain)
    out[totsamps - halfwid:totsamps + 1, :] = mat[totsamps - halfwid:totsamps + 1, :] * gain
    return out


def main():
    data_csv = os.path.join(BASE, "sample_data", "A8-NEW-1.csv")
    out_dir = os.path.join(BASE, "output")
    os.makedirs(out_dir, exist_ok=True)

    matrix_csv = os.path.join(out_dir, "A8-NEW-1_matrix.csv")
    ensure_matrix(data_csv, matrix_csv, nsamp=801)

    # parameters
    length_trace = 800
    start_position = 0
    scans_per_meter = 0.257058  # 1 / 3.89064

    # baseline: agcGain -> subtracting_average_2D
    agc_csv = os.path.join(out_dir, "A8-NEW-1_agcGain_win151.csv")
    agc_png = os.path.join(out_dir, "A8-NEW-1_agcGain_win151.png")
    agcGain(matrix_csv, agc_csv, agc_png, length_trace, start_position, scans_per_meter, window=151)

    base_csv = os.path.join(out_dir, "A8-NEW-1_baseline_subavg.csv")
    base_png = os.path.join(out_dir, "A8-NEW-1_baseline_subavg.png")
    subtracting_average_2D(agc_csv, base_csv, base_png, length_trace, start_position, scans_per_meter, ntraces=79)

    # background suppression improvement: SVD low-rank removal on AGC output
    agc_mat = readcsv(agc_csv)
    bg_imp = svd_lowrank_remove(agc_mat, rank=1)
    bg_imp_csv = os.path.join(out_dir, "A8-NEW-1_bg_svd_rank1.csv")
    bg_imp_png = os.path.join(out_dir, "A8-NEW-1_bg_svd_rank1.png")
    savecsv(bg_imp, bg_imp_csv)
    save_image(bg_imp, bg_imp_png, "Data[SVD low-rank removed]",
               time_range=(0, length_trace), distance_range=(start_position, start_position + bg_imp.shape[1] / scans_per_meter))

    # AGC improvement: larger window + gain cap, then subtracting_average_2D
    mat = readcsv(matrix_csv)
    agc_imp = agc_with_cap(mat, window=301, max_gain=10.0)
    agc_imp_csv = os.path.join(out_dir, "A8-NEW-1_agc_win301_cap10.csv")
    savecsv(agc_imp, agc_imp_csv)
    agc_imp_png = os.path.join(out_dir, "A8-NEW-1_agc_win301_cap10.png")
    save_image(agc_imp, agc_imp_png, "Data[agc win301 cap10]",
               time_range=(0, length_trace), distance_range=(start_position, start_position + agc_imp.shape[1] / scans_per_meter))

    agc_imp_sub_csv = os.path.join(out_dir, "A8-NEW-1_agcimp_subavg.csv")
    agc_imp_sub_png = os.path.join(out_dir, "A8-NEW-1_agcimp_subavg.png")
    subtracting_average_2D(agc_imp_csv, agc_imp_sub_csv, agc_imp_sub_png, length_trace, start_position, scans_per_meter, ntraces=79)


if __name__ == "__main__":
    main()
