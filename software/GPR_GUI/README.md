# GPR GUI (Enhanced Prototype)

Tkinter GUI to load CSV, display B-scan, and apply both **original** processing methods from `PythonModule_core` and **researched** methods (SVD background, F-K filter, Hankel SVD, sliding average).

## Features
- Import CSV
- Display B-scan (matplotlib)
- Method list (original + researched)
- Per-method parameter inputs (window width, time, rank, etc.)
- Output CSV/PNG saved under `output/`

## Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib
- scipy
- tkinter (usually included with Python)

Install deps:
```bash
pip install numpy pandas matplotlib scipy
```

## Run
From repo root:
```bash
python app.py
```

## Sample data
- Example B-scan CSV: `sample_data/sample_bscan.csv`

How to verify:
1) Run `python app.py`
2) Click **Import CSV** and select `sample_data/sample_bscan.csv`
3) Select a method and set parameters
4) Click **Apply Selected Method** to see output

## Repo layout
- `app.py` — main GUI (enhanced)
- `read_file_data.py` — minimal CSV IO helpers
- `output/` — generated results
