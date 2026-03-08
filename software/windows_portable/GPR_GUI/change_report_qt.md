# GPR GUI Tkinter → PyQt6 (Isolated)

## What changed
- Added **PyQt6** UI entrypoint: `app_qt.py` (new GUI layout + matplotlib Qt canvas)
- Preserved core workflow: CSV import → method selection → parameter inputs → B-scan render → batch compare → report
- Synced Tk additions: Undo/Reset, display downsample, colorbar/grid toggles, report logging
- Integrated themed UI support (qt-material preferred, qdarkstyle fallback)
- Kept existing processing logic and output format (CSV/PNG/report MD)
- Added dependency list: `dependencies_qt.txt`
- Added mock UI screenshot: `isolated/screenshots/gpr_gui_qt_mock.png`

## Notes
- PyQt version uses QtAgg backend and FigureCanvasQTAgg for plotting.
- `read_file_data.py` imported via fallback search path (repos/GPR_GUI or repos/PythonModule).

## Risks / Edge Cases
- **Dependency availability**: PyQt6/qt-material/qdarkstyle must be installed for full theme. Falls back to default if missing.
- **Core modules path**: `PythonModule_core` path is resolved via fallback. If repos are moved, adjust `CORE_DIR_CANDIDATES`.
- **Mock screenshot**: current screenshot is a schematic mock (no live GUI). Needs real capture on a machine with GUI.
- **Large datasets**: Matplotlib UI rendering may slow on very large arrays; existing downsample switches remain.

## How to run
```bash
python3 app_qt.py
```
