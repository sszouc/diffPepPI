import os
import torch

SEED = 42
MODEL_SAVE_DIR = "checkpoints"

# ---------------------------------------------------------------- 数据（Propedia 26 版）
DATA_DIR = "data"
DATA_CSV = os.path.join(DATA_DIR, "propedia26_clean.csv")      # 复合物标注表
RESIDUE_EMB_DIR = os.path.join(DATA_DIR, "residue_emb")        # 逐蛋白残基级表征 {hash}.npy
STRUCT_CACHE_DIR = os.path.join(DATA_DIR, "struct_cache")      # 对齐后的结构缓存 {id}.npz
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.pkl")         # 对齐质量 + seq100_clusters

# 训练/验证划分比例。划分以「连通分量」为单位：
#   PROTEIN_SEQ 相同 或 seq100_clusters 相同 的条目一定落在同一侧（肽级+蛋白级无泄漏）
SPLIT_RATIO = 0.9

# 蛋白序列里的未知残基（= 结构里缺坐标的位置）
UNKNOWN_PROTEIN_TOKEN = "X"

# ---------------------------------------------------------------- 词表
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
EOS_IDX = 20
PAD_IDX = 21
MASK_IDX = 22
X_IDX = 23
VOCAB_SIZE = 24
OUT_CLASSES = 21

PEPTIDE_LENGTH = 51

# ---------------------------------------------------------------- 扩散
T_STEPS = 100

TRAIN_T_STEPS = T_STEPS

# 验证用的时间步。不含 t=0：掩码率为 0 时没有加噪位置，只对加噪位置算的损失无法定义。
VAL_T_STEPS = (25, 50, 75, 99)

# ---------------------------------------------------------------- 模型
D_MODEL = 256
NUM_LAYERS = 4
NHEAD = 8
DROPOUT = 0.2
EMBEDDING_DIM = 1280

# Transformer FFN 宽度。保持 nn.TransformerDecoderLayer 的默认 2048（= 8×d_model）。
# 曾试过压到 1024（省 2.10 M 参数）：过拟合差距只降 9.4%，但验证损失 2.3700->2.4144、
# 生成 KL 0.0074->0.0094，且"打乱蛋白条件"的损失增幅从 +18.59% 掉到 +12.42%
# （= 对蛋白条件的利用能力损失三分之一）。详见 README「参数规模实验」。
FFN_DIM = 2048

# 长度分支隐层宽度。该子问题相对边际基线只能提升约 0.07 nats。
# 实测把 256 压到 64 会开始输出越界长度（3415 条里出现 0/35/41/49），
# 故保持 256。
LEN_HIDDEN = 256

# ---------------------------------------------------------------- 结构编码器
# 进结构分支前的残基长度上限。蛋白长度 P50=237 / P90=359 / P95=429，
# 截到 512 只影响 3.2% 的复合物（C 端尾部会被丢）。
STRUCT_MAX_LEN = 512
STRUCT_HIDDEN = 128          # EGNN 隐藏维
STRUCT_OUT = 256             # 输出给交叉注意力的维度（= D_MODEL）
STRUCT_LAYERS = 4            # EGNN 层数
STRUCT_KNN = 16              # Cα kNN 邻居数
STRUCT_RBF = 8               # 距离 RBF 编码的格点数

# 是否把「全局 ESM mean-pooling」作为一个额外的 memory token（双通道条件）
USE_GLOBAL_TOKEN = True

# 辅助界面预测损失的权重（0 表示关掉）
IFACE_LOSS_WEIGHT = 0.5

# ---------------------------------------------------------------- 训练
# 注意：加上结构编码器后显存占用与 batch 大小是超线性的。
# 实测（RTX 4060 Laptop, 8.6GB）：
#   batch=8  -> 1.05 GB, 63 ms/batch
#   batch=16 -> 1.97 GB, 105 ms/batch
#   batch=32 -> 3.76 GB, 193 ms/batch
#   batch=64 -> 8.82 GB（超过显存，溢到内存，慢 9 倍）
# 所以默认取 32。
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 500
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4

# 梯度裁剪上限（结构编码器 + 去噪网络）
GRAD_CLIP = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
