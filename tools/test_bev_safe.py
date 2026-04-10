#!/usr/bin/env python3
"""Run LocSim eval using the current /root/autodl-tmp/ultralytics repo only."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/ultralytics")
TEST_BEV = ROOT / "test_bev.py"

# Force-register custom modules/classes used by trained checkpoints so torch.load() during eval
# never fails with "Can't get attribute ...".
from ultralytics.nn.modules.head import Pose26MLPRefine, Pose26Refine  # noqa: F401
from ultralytics.nn.modules.transformer import P3CrossScaleDeformAttn  # noqa: F401


def main() -> None:
    if not TEST_BEV.exists():
        raise FileNotFoundError(TEST_BEV)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    runpy.run_path(str(TEST_BEV), run_name="__main__")


if __name__ == "__main__":
    main()
