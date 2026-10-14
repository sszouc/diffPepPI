import os
import torch

SEED = 42
DATA_PATH = "Train_Sequences.fasta"
EMBEDDING_PATH = "protein_esm.pkl"
MODEL_SAVE_DIR = "checkpoints"

ONLY_POSITIVE = True

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
EOS_IDX = 20
PAD_IDX = 21
MASK_IDX = 22
X_IDX = 23
VOCAB_SIZE = 24
OUT_CLASSES = 21

PEPTIDE_LENGTH = 51

T_STEPS = 100

TRAIN_T_STEPS = T_STEPS

# 验证用的时间步。不含 t=0：掩码率为 0 时没有加噪位置，只对加噪位置算的损失无法定义。
VAL_T_STEPS = (25, 50, 75, 99)

D_MODEL = 256
NUM_LAYERS = 4
NHEAD = 8
DROPOUT = 0.2
EMBEDDING_DIM = 1280

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 500
TOTAL_STEPS = 15000
EPOCHS = 500
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
