"""Install the measured CUDA stack without upstream dependency-pin rewrites."""

import argparse
import shutil
import subprocess
from pathlib import Path

FISH_SHA = "541566932abbaebcb0a40e562a1f846738383f38"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--fish-source", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv and put it on PATH before running this script")
    sha = subprocess.check_output(
        ["git", "-C", str(args.fish_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != FISH_SHA:
        raise ValueError(f"Fish Speech must be checked out at {FISH_SHA}; found {sha}")
    python = args.venv.resolve() / "bin/python"
    if not python.exists():
        subprocess.run([uv, "venv", "--python", "3.12", str(args.venv)], check=True)
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-r",
            str(root / "requirements-runtime.lock"),
        ],
        check=True,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-e",
            str(args.fish_source.resolve()),
            "-e",
            str(root),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            "import torch, flashinfer, drama3.model, drama3.codec; "
            "print('Runtime imports OK', torch.__version__, flashinfer.__version__)",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
