from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DATA_DIR = Path(PROJECT_ROOT_DIR) / ".." / "data"
CONFIG_DIR = PROJECT_DATA_DIR / "configs"
MODEL_DIR = PROJECT_DATA_DIR / "models"
