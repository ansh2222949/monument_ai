import os
from pathlib import Path

# ================= 📂 PROJECT STRUCTURE =================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ================= 💾 MODEL OUTPUT =================
SAVE_DIR = BASE_DIR / "outputs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = str(SAVE_DIR / "best_monument_model.h5")

# ================= 🧠 HYPERPARAMETERS =================


IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3

# ================= 🏛️ MONUMENT CLASSES =================

CLASSES = [
    "Angkor_Wat",
    "Chichen_Itza",
    "Christ_the_Redeemer",
    "Colosseum",
    "Great_Wall",
    "Machu_Picchu",
    "Petra",
    "Taj_Mahal"
]

NUM_CLASSES = len(CLASSES)
