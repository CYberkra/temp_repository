# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app_qt.py'],
    pathex=['E:\\Openclaw\\.openclaw\\workspace\\repos\\PythonModule_core'],
    binaries=[],
    datas=[('assets', 'assets'), ('read_file_data.py', '.')],
    hiddenimports=['compensatingGain', 'dewow', 'set_zero_time', 'agcGain', 'subtracting_average_2D', 'running_average_2D'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GPR_GUI_Qt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
