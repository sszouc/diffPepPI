#config.py
"""全局配置：路径、词表、扩散步数、网络与训练超参数。

本文件只声明常量，不做任何计算。项目的其它模块（data_utils / diffusion / model /
train / evaluate / generate）全部从这里 import 参数，因此调整实验设置（肽长度、
扩散步数、批量大小、学习率……）只需要改这一个文件。

【本次重做的改动】
    1. EMBEDDING_DIM 原先为 2560（对应 ESM2-3B / esm2_t36_3B_UR50D），
       现改为 1280（对应 ESM2-650M / esm2_t33_650M_UR50D），以降低算力需求；
       条件嵌入文件相应换成 protein_esm.pkl（5688 条蛋白 × 1280 维 float32）。
    2. 数据只取正样本（ONLY_POSITIVE），负样本不再混进训练目标。
    3. 词表扩展：新增 <X>（未知氨基酸，只作输入、不计损失）与 <EOS>（肽结尾），
       索引重排为 0..19 氨基酸 / 20 EOS / 21 PAD / 22 MASK / 23 X，
       输出类别变成 21 = 20 氨基酸 + EOS。
    4. 取消 25 长度上限：PEPTIDE_LENGTH 变为最大长度 51（数据最长 50 + EOS），
       短肽补 PAD、长肽不再截断，生成时靠 EOS 自然结束（变长生成）。
"""
import os
import torch

# ---------------- 路径与复现性 ----------------
SEED = 42                               # 全局随机种子：数据划分、模型初始化、随机采样 t 都用它
DATA_PATH = "Train_Sequences.fasta"      # 训练数据：类 FASTA 文本，每条记录含 Peptide:/Protein: 两行
EMBEDDING_PATH = "protein_esm.pkl"       # 预计算蛋白嵌入：pickle 字典 {蛋白序列(str): numpy(1280,) float32}
MODEL_SAVE_DIR = "checkpoints"           # 权重保存目录（文件末尾会自动创建）

# ---------------- 数据筛选 ----------------
# Train_Sequences.fasta 里其实有 8622 条 >positive_*（真结合肽）和 8622 条 >negative_*（非结合肽）。
# 本项目的任务定义是“给定蛋白 → 生成结合肽”，即建模 p(肽 | 蛋白, 结合)；
# 而扩散模型是密度模型，训练目标就是复现训练集的样本分布——把负样本一起喂进去，
# 等于让模型去拟合“结合肽与非结合肽的混合分布”，条件信号会被稀释。
# 因此这里默认只保留 >positive_* 记录（8622 条），负样本留给评估阶段做对照/判别使用。
ONLY_POSITIVE = True                     # True = 只用正样本训练与统计；False = 正负样本都用（旧行为）

# ---------------- 词表与序列长度 ----------------
# 索引布局（**顺序很重要**）：
#   0..19  20 种标准氨基酸（AMINO_ACIDS 的下标）
#   20     <EOS>  肽结束标记：既是输入 token，也是网络的第 21 个输出类别
#   21     <PAD>  补齐用；不参与损失，也在自注意力里被屏蔽
#   22     <MASK> 吸收态扩散的掩码标记
#   23     <X>    未知氨基酸：**只作为输入的 token**，永远不是输出类别、也不参与损失
# 之所以把 EOS 放在 20（紧跟 20 种氨基酸）而把 PAD/MASK/X 放到后面：
# 这样“输出类别索引”与“token 索引”在 0..20 上完全对齐，
# 采样时 torch.multinomial 得到的下标可以直接当作 token 写回序列，不需要任何重映射。
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"     # 20 种标准氨基酸；字符串下标即类别索引 0..19
EOS_IDX = 20                             # <EOS>：肽的真实结尾，训练时放在肽末尾，采样到它即停止解码
PAD_IDX = 21                             # <PAD>：长度不足 L_max 时补在末端，损失与注意力都屏蔽
MASK_IDX = 22                            # <MASK>：吸收态扩散的掩码标记，只在前向加噪中出现
X_IDX = 23                               # <X>：未知氨基酸（数据里的 'X'），输入可见但不计损失
VOCAB_SIZE = 24                          # 嵌入表大小 = 20 氨基酸 + EOS + PAD + MASK + X
OUT_CLASSES = 21                         # 网络输出维度 = 20 种氨基酸 + EOS（PAD/MASK/X 不做预测）
# 肽的最大长度 L_max。数据里最长 50 个残基，再加上末尾的 <EOS> 正好 51；
# 不足 L_max 的部分用 <PAD> 补齐。真实长度不再被强行截断到 25。
PEPTIDE_LENGTH = 51

# ---------------- 长度预测（先预测长度，再按该长度生成）----------------
# 思路：掩码扩散是并行、双向去噪的，模型没法像自回归模型那样"生成到 EOS 自然停"；
# 所以长度由**单独的长度预测器**从蛋白嵌入给出，生成时窗口长度就按它来定：
#   位置 0..k-1 放真实残基（需要采样），位置 k 放 <EOS>（长度已定，直接放终止符），
#   位置 k+1.. 放 <PAD>（不参与采样）。
# 这样 EOS 永远不会被"自由采样"出来，也就不会出现文献里的 <eos> overflow（过早终止）。
#
# 训练方式（train.py 里的两阶段，互不干扰，各自一个 AdamW 优化器）：
#   ① 长度分支：cond → 长度分布 → 与真实长度算交叉熵 → 只更新长度预测器
#   ② 肽分支  ：(x_t, t, cond) → 扩散交叉熵 → 只更新去噪网络
# 注意：**长度不作为去噪网络的输入**。训练输入 x_t 里本来就能读出真实长度
# （位置 L 是 <EOS>、之后是永不被掩码的 <PAD>，还带 padding mask），
# 实测把它当条件喂进去几乎不影响损失（换真值/预测值只差 0.002），等于没用上。
# 长度预测器的输出只在生成时用：决定窗口形状（采样到第几位、EOS 放哪、哪里是 PAD）。
# 两个损失不做加权求和，因此这里不需要"长度损失权重"这个常量。

# ---------------- 离散扩散（吸收态 / masked diffusion）----------------
T_STEPS = 100                            # 扩散总步数 T；掩码概率 p_mask = t / (T - 1)，故 t=T-1 时全掩码
T_SAMPLE_MIN = 0                        # 训练时随机采样时间步的下界（含）→ p_mask ≈ 0.10
T_SAMPLE_MAX = 99                       # 训练时随机采样时间步的上界（含）→ p_mask ≈ 0.90
                                         # 即 t ~ Uniform{50, 51, ..., 450}，中段噪声占主导

# ---------------- 去噪网络结构 ----------------
D_MODEL = 256                            # Transformer 隐层维度（token / 位置 / 时间嵌入统一到该维度）
NUM_LAYERS = 4                           # nn.TransformerDecoder 的层数
NHEAD = 8                                # 多头注意力头数（自注意力与交叉注意力相同）
DROPOUT = 0.1                            # dropout 概率
MEMORY_LENGTH = 16                       # 交叉注意力记忆 token 数：1 个蛋白向量复制成 16 个 memory token
EMBEDDING_DIM = 1280                     # 蛋白嵌入维度 = ESM2-650M 最后一层隐藏状态的均值池化结果；
                                         # 必须与实际 pkl 中向量长度一致（650M=1280，3B=2560）

# ---------------- 训练超参数 ----------------
BATCH_SIZE = 64                          # batch size（DataLoader 每次取多少样本），也是有效 batch
LEARNING_RATE = 1e-4                     # AdamW 学习率（全程恒定，没有调度器）
WEIGHT_DECAY = 1e-5                      # AdamW 权重衰减
WARMUP_STEPS = 500                       # 【注意】为学习率预热预留的常量，train.py 目前没有实现调度器，未被使用
TOTAL_STEPS = 15000                      # 【注意】同上，为调度器预留，当前未被使用
EPOCHS = 100                             # 训练轮数；train.py 会跑到 100 轮，没有早停

# ---------------- 设备 ----------------
# 有 NVIDIA GPU 就用 GPU，否则回落 CPU —— 这样别人克隆下来不用改代码就能跑
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import 本文件时就创建权重目录，避免训练结束保存权重时才发现目录不存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
