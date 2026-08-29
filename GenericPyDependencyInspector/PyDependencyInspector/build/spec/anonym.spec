# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['anonym.__main__'],
    pathex=[],
    binaries=[

    ],
    datas=[

    ],
    hiddenimports=[

    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='anonym',
    debug=False,
    strip=False,
    upx=True,
    console=True,
)