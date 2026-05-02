import os

VERSION = "v1"

RAW_PATH        = r"D:\EEG-Sleep-GNN\data\raw\sleep-edf-database-expanded-1.0.0\sleep-cassette"

GRAPH_PATH      = r"D:\EEG-Sleep-GNN\graphs\processed"
MODEL_DIR       = r"D:\EEG-Sleep-GNN\outputs\models"
LOG_PATH        = r"D:\EEG-Sleep-GNN\outputs\train_log.txt"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"best_model_{VERSION}.pt")

CHANNELS       = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
EPOCH_DURATION = 30
FS             = 100
MAX_GRAPHS     = None

BANDS = [
    ("delta",   0.5,  4.0),
    ("theta",   4.0,  8.0),
    ("alpha",   8.0, 13.0),
    ("spindle", 12.0, 15.0),
    ("beta",    13.0, 30.0),
]

CORR_THRESHOLD = 0.3
NUM_CLASSES    = 5
STAGE_NAMES    = ["W", "N1", "N2", "N3", "REM"]

# 3 nodes, 12 features each (10 base + 2 EOG-specific)
NODE_FEAT_DIM = 12
N_NODES       = 3

BATCH_SIZE   = 32
EPOCHS       = 150
LR           = 3e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO  = 0.8
HIDDEN       = 128
HEADS        = 4
DROPOUT      = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)