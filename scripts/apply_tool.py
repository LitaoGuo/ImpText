#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imptext.tools import ToolsWrapper


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply one ImpText image enhancement tool.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    wrapper = ToolsWrapper()
    wrapper.apply_tool(args.tool, args.image, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

