DATASET = "data/era5-t2m-5deg.zarr"

HISTORY = 8
FORECAST = 12

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

DEVICE = "cpu"

# CHECKPOINT_FILE = "data/t2m_cpu.pt"
CHECKPOINT_FILE = "data/t2m_cnn.pt"
LOG_FILE = "logs/train_cnn.log"
