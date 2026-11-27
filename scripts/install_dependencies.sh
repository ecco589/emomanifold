#!/bin/bash
# Install the Python dependencies listed in requirements.txt.
cd "$(dirname "$0")/.."
python3 -m pip install -r requirements.txt
