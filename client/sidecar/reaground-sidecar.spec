from PyInstaller.utils.hooks import collect_submodules

hidden = collect_submodules("transformers.models.qwen2_5_vl")
a = Analysis(
    ["app.py"],
    pathex=["..", "../.."],
    binaries=[],
    datas=[("../models.json", ".")],
    hiddenimports=["protocol", "infer.parse"] + hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6", "PyQt5", "PyQt6", "tkinter", "tensorflow", "tensorboard",
        "keras", "matplotlib", "pandas", "scipy", "sklearn", "datasets", "h5py",
        "cv2", "nltk", "torchaudio", "torchvision", "IPython", "jupyter",
        "notebook", "nbformat", "pytest", "py", "zmq", "openpyxl",
        "pyarrow", "lxml", "black", "win32com", "pythoncom",
        "pywintypes", "torch.testing._internal.optests",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="reaground-sidecar", debug=False,
    bootloader_ignore_signals=False, strip=False, upx=False,
    console=True, disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=False,
    upx_exclude=[], name="reaground-sidecar",
)
