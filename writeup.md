# CS336 作业 1（basics）：从零构建 Transformer 语言模型

**Writeup** — 简章版本 26.0.3（2026 春季）

- 姓名 / SUNet ID：
- 日期：

> 所有书面题目的答案写在下面的「答：」处。
> 曲线图统一存到 `figs/` 目录并按要求嵌入，例如 `![sweep](figs/lr_sweep.png)`。
> 仅需代码的题目以 checklist 形式列出，完成一项打勾即可。

---

## 2. Byte-Pair Encoding (BPE) Tokenizer（字节对编码分词器）

### Problem (unicode1): Understanding Unicode（理解 Unicode）— 1 分

**(a)** `chr(0)` 返回的是哪个 Unicode 字符？

**答：**'\x00'

**(b)** 该字符的字符串表示（`__repr__()`）与打印表示（`print`）有什么区别？

**答：**一个能显示一个不显示

**(c)** 当这个字符出现在文本中时会发生什么？

**答：**就是啥也没有

### Problem (unicode2): Unicode Encodings（Unicode 编码）— 3 分

**(a)** 相比 UTF-16 或 UTF-32，为什么更倾向于在 UTF-8 编码的字节上训练 tokenizer？可以对比几种编码在不同字符串上的输出。

**答：**因为是单字节，0-255很小，方便处理

**(b)** 简章中给出的 `decode_utf8_bytes_to_str_wrong` 函数为什么是错的？请给出一个会得到错误结果的输入字节串。

**答：**因为把解码一个字符和解码一个字节当成了一件事，“汉”

**(c)** 给出一个无法解码为任何 Unicode 字符的两字节序列。

**答：**b'\x80\x80'

### Problem (train_bpe): BPE Tokenizer Training（BPE 训练算法）— 15 分

- [ ] 实现 `train_bpe`，通过提供的测试

### Problem (train_bpe_tinystories): BPE Training on TinyStories（在 TinyStories 上训练 BPE）— 2 分

在 TinyStories 上训练 byte-level BPE，最大词表 10,000，加入 `<|endoftext|>` 特殊 token；把 vocab 和 merges 序列化到磁盘。

**(a)** 训练用了多少时间和内存？词表中最长的 token 是什么？它合理吗？

**答：**

**(b)** Profile 你的代码：训练过程中哪一步最耗时？

**答：**

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText（在 OWT 上训练 BPE）— 2 分

在 OpenWebText 上训练 byte-level BPE，最大词表 32,000；序列化 vocab 和 merges。

**(a)** 词表中最长的 token 是什么？它合理吗？

**答：**

**(b)** 对比在 TinyStories 和 OpenWebText 上训练出的两个 tokenizer 的异同。

**答：**

### Problem (tokenizer): Implementing the tokenizer（实现分词器）— 15 分

- [ ] `Tokenizer` 类（encode / decode / 从 vocab+merges 构建）实现完成，通过 `tests/test_tokenizer.py`

### Problem (tokenizer_experiments): Experiments with tokenizers（分词器实验）— 4 分

**(a)** 从 TinyStories 和 OpenWebText 各采样 10 篇文档，用对应 tokenizer（10K / 32K 词表）编码。各自的压缩率（bytes/token）是多少？

**答：**

**(b)** 如果用 TinyStories tokenizer 去编码 OpenWebText 样本会发生什么？比较压缩率并定性描述。

**答：**

**(c)** 估计 tokenizer 的吞吐量（bytes/秒）。tokenize Pile 数据集（825GB 文本）需要多久？

**答：**

**(d)** 把数据集编码成 token ID 数组时用 `uint16` 存储，为什么这是一个合适的选择？

**答：**

---

## 3. Transformer Language Model Architecture（Transformer 语言模型架构）

### 代码题 checklist

- [ ] (linear) Linear — 1 分
- [ ] (embedding) Embedding — 1 分
- [ ] (rmsnorm) RMSNorm — 1 分
- [ ] (positionwise_feedforward) SwiGLU FFN — 2 分
- [ ] (rope) `RotaryPositionalEmbedding` — 2 分
- [ ] (softmax) softmax — 1 分
- [ ] (scaled_dot_product_attention) 缩放点积注意力 — 5 分
- [ ] (multihead_self_attention) 因果多头自注意力 — 5 分
- [ ] (transformer_block) Transformer block — 3 分
- [ ] (transformer_lm) Transformer LM — 3 分

### Problem (transformer_accounting): Transformer LM resource accounting（资源核算）— 5 分

GPT-2 XL 配置：`vocab_size=50257`、`context_length=1024`、`num_layers=48`、`d_model=1600`、`num_heads=25`、`d_ff=4288`。

**(a)** 可训练参数有多少？按 fp32 计算，仅加载模型需要多少显存？

**答：**

**(b)** 列出一次前向传播涉及的所有矩阵乘法；总 FLOPs 是多少（输入为 context_length 个 token）？

**答：**

**(c)** 模型中哪部分占的 FLOPs 最多？

**答：**

**(d)** 对 GPT-2 small（12 层，768，12 头）、medium（24，1024，16）、large（36，1280，20）重复上述分析：给出各组件 FLOPs 占单次前向总 FLOPs 的比例；模型变大时各部分比例如何变化？

**答：**

**(e)** 把 GPT-2 XL 的上下文长度提高到 16,384：单次前向的总 FLOPs 如何变化？各组件的相对贡献如何变化？

**答：**

---

## 4. Training a Transformer LM（训练组件）

### 代码题 checklist

- [ ] (cross_entropy) 交叉熵损失 — 1 分
- [ ] (adamw) AdamW — 2 分
- [ ] (learning_rate_schedule) 带 warmup 的余弦学习率调度 — 1 分
- [ ] (gradient_clipping) 梯度裁剪 — 1 分
- [ ] (data_loading) 数据加载 — 2 分
- [ ] (checkpointing) checkpoint 保存/加载 — 1 分
- [ ] (training_together) 完整训练脚本 — 4 分
- [ ] (decoding) 解码（生成）— 3 分

### Problem (learning_rate_tuning): Tuning the learning rate（调学习率）— 1 分

玩具 SGD 例子分别用 lr = 1e1、1e2、1e3 跑 10 步。三种情况下 loss 分别是什么表现（下降更快/更慢/发散）？

**答：**

### Problem (adamw_accounting): Resource accounting for training with AdamW（AdamW 资源核算）— 2 分

所有张量按 fp32 计算；假设 `d_ff = 8/3 · d_model`。

**(a)** 用 batch_size 和模型超参（vocab_size、context_length、num_layers、d_model、num_heads）表示 AdamW 训练的峰值显存，分别拆解为参数、激活值、梯度、优化器状态（激活值只考虑简章列出的组件）。

**答：**

**(b)** 代入 GPT-2 XL 配置：得到形如 `a·batch_size + b` 的表达式，并给出 80GB 显存下能用的最大 batch size。

**答：**

**(c)** 一步 AdamW 需要多少 FLOPs？（代数表达式 + 简要理由）

**答：**

**(d)** MFU：H100 理论峰值 495 TFLOPS（fp32/TF32），按 50% MFU 计算，GPT-2 XL、400K 步、batch 1024（反向 FLOPs = 2× 前向）需要训练多少小时？

**答：**

---

## 7. Experiments（实验）

### Problem (experiment_log): Experiment logging（实验记录）— 3 分

- [ ] 日志基础设施（按梯度步 + 墙钟时间记录 loss 曲线）
- [ ] 实验日志文档（记录本节所有实验）

### Problem (learning_rate): Tune the learning rate（调学习率）— 3 分

**(a)** 学习率扫描：多条学习率对应的 loss 曲线 + 说明你的搜索策略。目标：TinyStories 上每 token 验证 loss ≤ 1.45（low-resource 配置 40M tokens 时为 ≤ 2.00）。

**答：**

![lr sweep](figs/lr_sweep.png)

**(b)** "稳定边缘"（edge of stability）：做出包含至少一条发散 run 的递增学习率曲线，并分析发散点与最优学习率的关系。

**答：**

![edge of stability](figs/lr_edge_of_stability.png)

### Problem (batch_size_experiment): Batch size variations（batch size 变化）— 1 分

batch size 从 1 取到显存上限，中间至少取几个值（包括 64、128），必要时重新调 lr。

**答（曲线）：**

![batch sizes](figs/batch_sizes.png)

**答（讨论）：**

### Problem (generate): Generate text（生成文本）— 1 分

至少 256 个 token 的生成文本（或直到第一个 `<|endoftext|>`），+ 对流畅度的简短评论和至少两个影响输出质量的因素。

**答：**

```
（把生成的文本粘到这里）
```

**评论：**

### Problem (layer_norm_ablation): Remove RMSNorm and train（去掉 RMSNorm）— 1 分

**(a)** 用之前的最优 lr 去掉 RMSNorm 训练的学习曲线，以及最佳（更低？）lr 下的曲线。

**答：**

![no rmsnorm](figs/ablation_no_rmsnorm.png)

**(b)** 对 RMSNorm 影响的几句话评论。

**答：**

### Problem (pre_norm_ablation): Implement post-norm and train（post-norm 实验）— 1 分

**答（post-norm 与 pre-norm 的 loss 曲线对比）：**

![post-norm vs pre-norm](figs/ablation_post_norm.png)

### Problem (no_pos_emb): Implement NoPE（去掉位置编码）— 1 分

**答（RoPE 与 NoPE 的 loss 曲线对比）：**

![rope vs nope](figs/ablation_nope.png)

### Problem (swiglu_ablation): SwiGLU vs. SiLU — 1 分

SiLU 基线用 `d_ff = 4 × d_model`，以近似对齐参数量。

**答（曲线）：**

![swiglu vs silu](figs/ablation_swiglu.png)

**答（讨论）：**

### Problem (main_experiment): Experiment on OWT（OpenWebText 实验）— 2 分

**(a)** 在 OpenWebText 上训练的 loss 曲线；描述与 TinyStories 的 loss 差异，以及如何解读。

**答：**

![owt curve](figs/owt_learning_curve.png)

**(b)** OWT 模型生成的文本（格式同 TinyStories）+ 流畅度评论；同样模型和计算量下为什么质量更差？

**答：**

```
（把生成的文本粘到这里）
```

**评论：**

### Problem (leaderboard): Leaderboard（排行榜）— 6 分

最终验证 loss + 一张墙钟时间 x 轴小于 45 分钟的学习曲线 + 你所做修改的描述。

**答：**

![leaderboard](figs/leaderboard.png)

---

## 附录：基础配置（来自简章）

| 设置 | 基础配置 | Low-resource（CPU/MPS） |
|---|---|---|
| 词表大小 | 10000 | 10000 |
| 上下文长度 | 256 | 256 |
| d_model | 512 | 512 |
| d_ff | 1344 | 1344 |
| RoPE theta | 10000 | 10000 |
| 层数 / 头数 | 4 / 16 | 4 / 16 |
| 总 token 数 | 327,680,000 | 40,000,000 |
| 目标验证 loss | ≤ 1.45 | ≤ 2.00 |

运行日志（边做边填）：

| Run | 题目 | lr | batch | steps | 备注 |
|---|---|---|---|---|---|
| | | | | | |
