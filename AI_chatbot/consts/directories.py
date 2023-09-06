from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DATA_DIR = ROOT_DIR / "intents"

DATA_DIR = ROOT_DIR / "data"

OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"