#!/usr/bin/env python3
"""Entry point for GPU Control Panel.

See README.md, or just run ./run.sh from this directory.

Usage:
    .venv/bin/python main.py [--debug]

--debug shows the discovered hwmon path as a startup toast.
"""
import sys

from gpuctl.app import main

if __name__ == "__main__":
    main(debug="--debug" in sys.argv)
