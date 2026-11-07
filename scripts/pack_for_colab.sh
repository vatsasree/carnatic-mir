#!/usr/bin/env bash
# Package the code (no audio, outputs or model weights) for upload to Google Drive -> MyDrive/carnatic_mir/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/carnatic_mir_code.zip
zip -qr dist/carnatic_mir_code.zip carnatic_mir app scripts colab requirements.txt requirements-colab.txt README.md urls_example.txt \
    -x '*__pycache__*' '*.pyc' '.DS_Store'
ls -lh dist/carnatic_mir_code.zip
