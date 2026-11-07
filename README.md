# 基于条件离散扩散模型，结合达蛋白质3D结构和口袋接触残基的蛋白质结合肽生成

给定一条蛋白质的序列、PDB解析的三维结构和蛋白质-肽口袋接触残基，使用掩码式离散扩散模型（masked diffusion），结合ESM2-650M注入条件向量与EGNN学习PDB结构，预测口袋接触残基并生成与它结合的多肽序列。
技术栈：PyTorch、Transformer、离散扩散、EGNN等

<img width="1837" height="1082" alt="image" src="https://github.com/user-attachments/assets/b5bd9161-b80e-48b6-a20c-11f3d630aa58" />

该图是 PDB 编号为 8D5N-B-A 的已知蛋白质-肽天然复合物，其中肽序列为`HPGSVNEFDFGC`，AF3预测其ipTM = 0.95 pTM = 0.67

<img width="1940" height="1104" alt="image" src="https://github.com/user-attachments/assets/32659b0c-a329-4ad7-bffa-165e2a6f1fe8" />

该图是模型根据 8D5N-B-A 蛋白质，生成的结合肽，由此得到的蛋白质-肽复合物，其中肽序列为`HPPSSFRFM`，AF3预测其ipTM = 0.93 pTM = 0.67。可以看见，生成的结合肽与自然肽序列差异很大，但是 AF3 给出了高度类似的结果。说明模型的确捕获到了准确的信号。

## 1. 任务与数据

任务：输入蛋白质的表征，生成结合肽序列（长度 5~30），即建模 `p(肽 | 蛋白)`。

**数据来源：Propedia 26**（73,392 个蛋白–肽复合物中清洗出的高质量子集）。

清洗条件（见 [`data/sequence.py`](data/sequence.py)）：

- 肽长度 5~30（排除过短/过长）
- 肽序列不含未知氨基酸 `X`
- `BPP% >= 20`（肽的包藏嵌入比例，避免肽大部分游离在溶剂里）
- 分辨率 ≤ 3 Å（排除 NMR 及低分辨率结构）
- 只保留条目数 ≥ 50 的物种（避免罕见物种污染）

| 项目 | 内容 |
|---|---|
| 复合物条目 | 10813 |
| 唯一蛋白 | 5,926 |
| 唯一肽 | 4,825 |
| 肽长度 | 5~30，中位 11，均值 12.7 |
| 蛋白长度 | 51~2773，中位 237 |
| 界面残基 | 每复合物中位 25 个，占蛋白长度中位 12.2% |
| 数据划分 | 9,695 训练 / 1,078 验证 |


## 2. 方法

```text
蛋白序列 ──ESM2-650M──▶ 逐残基表征 [L,1280] ─┐
蛋白 3D 结构 (PDB) ──▶ CA 坐标 + 界面标注 ───┤
                                             ▼
                                  StructureEncoder (4 层 EGNN)
                                             │
                       ┌─────────────────────┼──────────────────────┐
                       ▼                     ▼                      ▼
                  memory (L+1,256)     iface_logits (L)        全局 ESM 均值
                  = 全局token + 残基     → 辅助 BCE 损失            → 长度预测器 → k
                       │
                       ▼
              去噪网络 (4 层 TransformerDecoder) ◀── 采样窗口 0..k-1
                       │
                T=100 步去噪 ──▶ 肽序列
```

### 2.1 掩码扩散过程

前向加噪：时间步 `t ∈ {0, ..., T-1}`，掩码率 `p_t = t/(T-1)`，
序列的每个位置以概率 `p_t` 被替换为 `<MASK>`；`<PAD>`、`<X>`、`<EOS>` 三类不参与加噪。
`<EOS>` 也排除的理由：它的位置由长度分支预先给定，采样时概率被置 0、不参与去噪；
若允许遮蔽，模型还能从未遮蔽的 `<EOS>` 直接读出长度，造成信息泄漏。

反向去噪：从全掩码出发，每一步只对被遮蔽的位置重新采样。对吸收态扩散，
反掩码概率有闭式解 `1 - p_{t-1} / p_t`，不需要额外学一个转移网络。实现见 `diffusion.p_sample_loop`。

训练损失（`train.masked_loss`）：把加噪后的 `x_t` 送进网络预测每个位置的原始 token，
交叉熵只对被加噪的位置计算，另外补上 `<EOS>` 位置：

- 未加噪的位置不计损失。模型看得见它们（实测准确率 100%、loss 仅 0.003），
  算进去只是把分母稀释掉；而且采样时只用被遮蔽位置的预测，未遮蔽位的输出根本用不到。
  实测 t=50 时未加噪位占有效位的 51%，旧口径会把损失拉低约 0.02。
- `<EOS>` 位置计入。它永不加噪，但必须学会"到此结束"；若不补这一项，
  `<EOS>` 在本设计下会完全失去监督。
- `<PAD>` 与 `<X>` 天然被排除：既不加噪、目标也不合法。

### 2.2 结构编码器（`struct_encoder.StructureEncoder`）

用蛋白 3D 结构 + 逐残基 ESM 表征，产出交叉注意力的 memory 和一个界面预测头。

| 组件 | 说明 |
|---|---|
| 节点特征 | 氨基酸 embedding(32) + 逐残基 ESM(1280→128) + `present`/`imputed` 两个标记 |
| 坐标 | CA 三维坐标；`<X>` 与末端缺口由 `node_mask` 屏蔽 |
| 图 | Cα kNN（k=16），边特征 = 距离的 8 点 RBF 展开 |
| 主干 | **4 层 EGNN**（E(n) 等变图网络，**无坐标更新**，见下） |
| 全局补足 | 除了EGNN，每个残基的输出再拼上全蛋白池化向量 |
| 输出 memory | `(B, L+1, 256)` = 1 个全局 ESM token + L 个残基表征 |
| 界面头 | 逐残基 1 维 logits，`BCE` 辅助损失（权重 `IFACE_LOSS_WEIGHT=0.5`） |

**为什么把界面标注当"辅助标签"而不是输入**：推理阶段面对的是没见过的新蛋白，此时不知道它的结合位点。把真实界面当选条件会造成输入在推理时不存在的泄漏。
所以让编码器看整条蛋白链、自己学口袋表征，界面标注只用来加一个辅助监督头。

**长度上限**：结构分支最多 512 个残基（`STRUCT_MAX_LEN`）。

### 2.3 去噪网络（`model.ConditionalDecoder`）

| 组件   | 说明                                                             |
|------|----------------------------------------------------------------|
| 输入   | `x_t`（B×51）、时间步 `t`、`memory`（B×(L+1)×256）、`memory_mask`         |
| 词嵌入  | 24 个 token 的嵌入表（20 氨基酸 + `<EOS>` + `<PAD>` + `<MASK>` + `<X>`） |
| 位置嵌入 | 可学习，51 个位置                                                     |
| 时间嵌入 | 正弦嵌入 + MLP，广播加到每个位置                                            |
| 主干   | 4 层 `nn.TransformerDecoder`，双向自注意力                             |
| 条件注入 | 交叉注意力：Q 来自肽，K/V 来自结构编码器的 `memory`                              |
| 输出   | 21 类打分（20 种氨基酸 + `<EOS>`）                                      |

去噪任务是双向的：每个位置都要能看到整条序列，因此不加因果掩码；
肽尾部的 `<PAD>` 通过 `tgt_key_padding_mask` 屏蔽，`memory` 的 padding 通过
`memory_key_padding_mask` 屏蔽。

**条件注入是变长的**：肽的 51 个位置各自对 `L+1` 个 memory 位置算注意力权重。

**结构编码只做一次**：蛋白条件与扩散时间步无关，所以每个 batch（无论训练、验证的
4 个时间步，还是采样的 100 步）都复用同一份 `memory`。这是采样可行的关键——
否则每步都要重跑 EGNN。

### 2.4 长度预测分支（`model.LengthPredictor`）

肽长不固定（5~30），因此单独用一个分支预测长度：

- 结构：MLP `1280 → 256 → 256 → 51`，输出 `k = 0..50` 的长度分布，交叉熵训练
- 独立训练：与去噪网络各用一个 AdamW 优化器，每个 batch 先更新长度分支再更新去噪分支
- 生成时采样长度：从预测分布采样 `k`（而不是取 argmax）。采样保留了分布形状，
  同一蛋白的多条候选肽长度也自然不同；argmax 会一直偏向众数
- 长度不作为网络输入：`k` 只用于构造采样窗口（`0..k-1` 待采样、第 `k` 位放 `<EOS>`、其余 `<PAD>`）

### 2.5 词表与特殊 token

| 索引   | token    | 作用               |
|------|----------|------------------|
| 0~19 | 20 种氨基酸  | 生成目标             |
| 20   | `<EOS>`  | 肽结束              |
| 21   | `<PAD>`  | 补齐到 51 位，不计损失    |
| 22   | `<MASK>` | 扩散过程的掩码态         |
| 23   | `<X>`    | 未知氨基酸，保留在输入但不计损失 |

## 3. 代码结构

```
config.py        全部超参数 + 数据路径（扩散、模型、结构编码器、训练、验证协议、早停、词表）
struct_encoder.py StructureEncoder：Cα kNN 图 + 4 层 EGNN -> memory + 界面 logits
data_utils.py    Propedia 数据管线：CSV/残基表征/结构缓存、编码解码、padding collate、无泄漏划分
diffusion.py     加噪 q_sample、反向采样 p_sample_loop
model.py         TimeEmbedding / ConditionalDecoder（消费变长 memory）/ LengthPredictor
train.py         训练主循环（结构编码 + 去噪 + 界面辅助损失，masked-only 损失，早停）
evaluate.py      批量生成与指标（KL / 多样性 / 完全匹配率 / 长度分布）
generate.py      单条生成示例
analyze_run.py   诊断模型：条件敏感性、结构贡献、界面预测 AUROC/AUPRC、过拟合差距
export_val_proteins.py  导出 val 侧 683 个蛋白（FASTA + CSV），供外部结构验证选靶

# 预处理（只需跑一次）
split_pkl.py     7.2GB 残基级表征 pkl -> data/residue_emb/{hash}.npy（逐蛋白，懒加载）
build_align.py   PDB -> data/struct_cache/{id}.npz（对齐到 PROTEIN_SEQ 位置）+ manifest.pkl
verify_alignment.py  全量验证所有索引是否一致（7 项检查）
```

**首次使用需按顺序跑两步预处理**（见第 4 节）：

1. `python build_align.py` —— 解析 10,813 个 PDB，产出结构缓存与对齐质量表（约 40 秒）
2. `python split_pkl.py` —— 拆分 7.2 GB 的残基表征 pkl（需先腾出 ≥ 8 GB 可用内存，约 5~10 分钟）

训练时每个 epoch 只写一个 `checkpoints/model_last.pt`，
并在验证损失刷新最优时另存一份 `checkpoints/model_best.pt`：
- **验证损失是确定性的**：固定在一组时间步 `VAL_T_STEPS = (25, 50, 75, 99)` 上算，
  掩码随机场由独立生成器（种子 = `SEED`**）给出。**
- **早停**：连续 `EARLY_STOP_PATIENCE = 10` 轮验证损失没有改善（改善幅度需超过
  `EARLY_STOP_MIN_DELTA = 1e-4`）就提前结束。


## 4. 环境与运行

```bash
pip install -r requirements.txt

# --- 预处理（只需一次）---
python build_align.py                  # PDB -> 结构缓存 + 对齐表（约 40 秒）
python split_pkl.py                    # 7.2GB 表征 pkl -> 逐蛋白 npy（需 >=8GB 可用内存）

# --- 训练与评估 ---
python train.py                                         # 训练（早停，最优权重存 checkpoints/）
python evaluate.py --split val --checkpoint checkpoints/model_best.pt
python generate.py --checkpoint checkpoints/model_best.pt
```

`build_align.py` 与 `split_pkl.py` 都是可重入的：已生成的产物会自动跳过，除非加 `--overwrite`。

## 5. 结果

在**训练时完全未见过的 683 个蛋白**上评估（按连通分量划分，蛋白级/肽级泄漏均为 0），
每个蛋白生成 5 条肽（共 3,415 条），长度由长度分支的预测分布采样，`seed=42`。
最佳权重：epoch 25，验证损失 **2.3700**（`VAL_T_STEPS=(25,50,75,99)` 上掩码位置的交叉熵均值）。

复现：`python evaluate.py --split val --checkpoint checkpoints/model_best.pt`

下表的指标都是分布层面的（生成肽与训练肽"像不像"）。5.3 另给出一组外部结构验证：
用 AlphaFold 3 预测复合物，检验生成肽是否被独立的第三方模型判为稳定界面。

| 指标 | 数值 | 说明 |
|---|---|---|
| KL(P_train ‖ P_gen) | 0.0074 | 生成肽与训练肽的氨基酸组成分布距离，越小越好 |
| 生成肽Diversity | 13.7379 | 全体候选肽的平均两两编辑距离 |
| 真实肽Diversity | 14.5187 | 全体已知肽的平均两两编辑距离 |
| 完全匹配率 | 4.13% | 与训练集肽完全相同（141/3,415），即新颖性 95.87% |
| 生成长度（均值 / 中位数） | 12.75 / 11.0 | 真实肽为 12.62 / 10.0 |
| 长度分布 TV 距离 | 0.1246 | 生成长度分布与真实长度分布的全变差距离 |
| 同一蛋白的长度极差 | 9.25 | 采样长度带来的候选多样性（取 argmax 时该值为 0） |


### 5.1 外部结构验证：用 AF3 检验生成肽能否形成稳定界面

前面的指标衡量的是"像不像真实肽"，回答不了"能不能结合"。为此补了一组验证：用 AlphaFold 3 预测 蛋白–肽 复合物结构，读取界面
置信度 **ipTM**（0–1，越高表示模型越确信两者形成稳定界面）。天然配体作为阳性对照。

选靶规则先定后跑：在 683 个留出蛋白中按界面残基数量与蛋白长度筛出候选，覆盖界面残基6–79 的整个范围
同一规则下的候选清单由 `export_val_proteins.py` 产出
（`val_proteins.csv` 含 `n_iface` / `protein_len` / `true_peptides` 等列）。
每个靶点 5 条生成肽 + 1 条天然配体作为阳性对照。

| 靶点 | 界面残基 | 天然配体 ipTM | 生成肽 ipTM（均值 / 最好） | 天然 − 最好生成 |
|---|---|---|---|---|
| 6BVV-B-A | 79 | 0.84 | 0.59 / 0.76 | +0.08 |
| 8D5N-B-A | 50 | 0.95 | 0.74 / 0.93 | +0.02 |
| 7SL5-I-G | 20 | 0.55 | 0.53 / 0.64 | −0.09 |
| 1U8H-C-A | 6 | 0.66 | 0.74 / 0.82 | −0.16 |

在界面明确的靶点上，排序与天然配体一致。6BVV（79 个界面残基）天然配体 0.84，5 条生成肽最好 0.76，全部低于天然配体；
8D5N（50 个）天然配体 0.95，最好一条 0.93，达到天然配体的 **98%**。生成肽被预测为与靶点形成稳定界面，强度接近天然配体。

20 条生成肽长度 5–12、组成均为疏水/极性混合，全部被 AF3 给出界面置信度（ipTM 0.38–0.93，中位数 0.66），没有出现无法折叠或明显不合理的输出；
pTM 与天然配体处于同一区间（0.66–0.91 vs 0.67–0.90）。这说明生成分布确实落在真实界面肽的流形附近，这与 5 节的 KL 0.0074 相互印证。


## 6. 参考

- D3PM: *Structured Denoising Diffusion Models in Discrete State-Spaces*, NeurIPS 2021
- MDLM: *Simple and Effective Masked Diffusion Language Models*, arXiv:2406.07524
- Eso-LMs: 掩码扩散语言模型中的长度建模, arXiv:2506.01928
