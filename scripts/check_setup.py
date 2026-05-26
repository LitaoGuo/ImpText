#!/usr/bin/env python3
import argparse
import importlib
import sys


REQUIRED_MODULES = ["openai", "PIL", "cv2", "numpy", "requests", "tqdm", "huggingface_hub"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether required ImpText release dependencies are importable.")
    parser.parse_args()

    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    failed = []
    for module in REQUIRED_MODULES:
        try:
            imported = importlib.import_module(module)
        except Exception as exc:
            failed.append((module, str(exc)))
            print(f"[FAIL] {module}: {exc}")
            continue
        version = getattr(imported, "__version__", "unknown")
        print(f"[OK] {module}: {version}")

    if failed:
        print("\nInstall missing dependencies with: pip install -r requirements.txt")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
