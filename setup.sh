#!/usr/bin/env bash
# LAB-2681 setup: one command, run from this folder.   bash setup.sh
#
# Your machine has Python 3.9; the lab needs 3.11 (Mellea requires it). `uv` downloads a private
# Python 3.11 into your home folder: no admin rights, nothing system-wide changes.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv (a fast Python package manager) for your user only"
  python3 -m pip install --user --quiet uv
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Creating the lab environment with Python 3.11 in ./.venv"
uv venv --python 3.11 .venv

echo "==> Installing Mellea, Qiskit and the notebook tools (a few minutes)"
uv pip install --python .venv/bin/python -r requirements.txt

echo "==> Registering the Jupyter kernel 'LAB-2681 (Mellea + Qiskit)'"
.venv/bin/python -m ipykernel install --user --name lab-2681 --display-name "LAB-2681 (Mellea + Qiskit)" >/dev/null

echo "==> Checking the install"
.venv/bin/python check_setup.py
