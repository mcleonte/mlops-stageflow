"""
Shared global variables for trainings modules.
"""

import os

from dotenv import load_dotenv
from pathlib import Path

# load .env file if dev mode
if os.environ["ENV_MODE"] in ("debug", "dev"):
  load_dotenv()

ENVS = os.environ.copy()

DATA_PATH = Path(ENVS["DATA_PATH"])
MODEL_PATH = Path(ENVS["MODEL_PATH"])
PREPROCESSORS_PATH = Path(ENVS["PREPROCESSORS_PATH"])

PATHS = {
    "data": DATA_PATH,
    "model": MODEL_PATH,
    "preprocessors": PREPROCESSORS_PATH,
    "raw": DATA_PATH / "complaints_processed.csv",
    "cleaned": DATA_PATH / "cleaned.csv",
    "train": DATA_PATH / "train.csv",
    "val": DATA_PATH / "val.csv",
    "test": DATA_PATH / "test.csv",
    "x_train": DATA_PATH / "x_train.joblib",
    "x_val": DATA_PATH / "x_val.joblib",
    "x_test": DATA_PATH / "x_test.joblib",
    "y_train": DATA_PATH / "y_train.csv",
    "y_val": DATA_PATH / "y_val.csv",
    "y_test": DATA_PATH / "y_test.csv",
    "label_encoder": PREPROCESSORS_PATH / "label_encoder.joblib",
    "vectorizer": PREPROCESSORS_PATH / "vectorizer.joblib",
    "selector": PREPROCESSORS_PATH / "selector.joblib"
}

for path in (MODEL_PATH, DATA_PATH, PREPROCESSORS_PATH):
  if not os.path.exists(path):
    os.makedirs(path)
