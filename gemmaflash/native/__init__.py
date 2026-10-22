"""Builds (once per source hash and MLX version) and imports the native expert gate."""

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CACHE = Path("~/.cache/gemmaflash/xgate").expanduser()


def load():
    import mlx.core  # noqa: F401  (registers mx.array with nanobind first)

    src = b"".join((HERE / f).read_bytes() for f in ("xgate.cpp", "CMakeLists.txt"))
    # The library links (by rpath) to this environment's libmlx: one build per MLX install.
    key = hashlib.sha256(src + mlx.core.__version__.encode() + mlx.core.__file__.encode()).hexdigest()[:16]
    out = CACHE / key
    so = next(out.glob("_xgate*.so"), None) if out.exists() else None
    if so is None:
        if shutil.which("cmake") is None:
            raise RuntimeError("building the native gate needs cmake (brew install cmake) and Xcode command line tools")
        build = out / "build"
        build.mkdir(parents=True, exist_ok=True)
        print("gemmaflash: building the native expert gate (once)...", file=sys.stderr, flush=True)
        subprocess.run(["cmake", "-S", str(HERE), "-B", str(build), f"-DPython_EXECUTABLE={sys.executable}",
                        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={out}"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["cmake", "--build", str(build), "-j8"], check=True, stdout=subprocess.DEVNULL)
        so = next(out.glob("_xgate*.so"))
    spec = importlib.util.spec_from_file_location("_xgate", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
