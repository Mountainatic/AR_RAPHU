#!/usr/bin/env python3
"""Single-process entrypoint for the frozen spectral suite."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("tools.run_spectral_suite", run_name="__main__")
