# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

# Pyinstallers dependency analysis will not detect that pylons TL DLLs are
# required (since they are searched and loaded from machine code at runtime).
# Consequently it would not include them in the archive. This is amended by
# simply adding all DLLs from the pypylon directory to the list of binaries.

import pypylon
import pathlib
pypylon_dir = pathlib.Path(pypylon.__file__).parent
pypylon_dlls = [(str(dll), '.') for dll in pypylon_dir.glob('*.dll')]
pypylon_pyds = [(str(dll), '.') for dll in pypylon_dir.glob('*.pyd')]

_binaries = list()
_binaries.extend(pypylon_dlls)
_binaries.extend(pypylon_pyds)

_pathex = list()
_pathex.append(str(pypylon_dir))

_hiddenimports = list()
_hiddenimports.extend(['pypylon', 'pypylon.pylon', 'pypylon.genicam', 'pypylon._pylon', 'pypylon._genicam'])


a = Analysis(
    ['main.py'],
    pathex=_pathex,
    binaries=_binaries,
    datas=[],
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='main',
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
