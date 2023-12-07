from pathlib import Path

SETTINGS_PATH = Path(__file__)
BENCHMARK_DIR = SETTINGS_PATH.parent
PROJECT_DIR = BENCHMARK_DIR.parent

BUILD_DIR = PROJECT_DIR / "build"

LIBCUECC_SO_PATH = BUILD_DIR / "libcuecc.so"
