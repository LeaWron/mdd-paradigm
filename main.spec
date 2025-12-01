# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['psycho\\main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('psycho/conf', 'psycho/conf'),
        ('psycho/stims', 'psycho/stims')
    ],
    hiddenimports=['hydra',
        'hydra.core',
        'hydra.core.config_store',
        'hydra._internal',
        'hydra.main',
        'omegaconf',
        'yaml',
        'psychopy',
        'psychopy.core',
    'psychopy.visual',
    'psychopy.event',
    'psychopy.data',
    'psychopy.hardware',
    'psychopy.sound',
    'psychopy.tools',
    'psychopy.constants'],
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
    name='psycho',
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
