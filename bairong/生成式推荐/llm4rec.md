
# 一、 LLM 作为特征工程 / Embedding

在这种模式下，LLM 充当“知识提取器”，不直接参与预测，而是通过理解非结构化数据来丰富传统模型（如 MMoE、DeepFM）的输入。

---

## **方案 A：文本摘要与标签提取（Text Profiling）**

利用 LLM 将冗长的产品描述、评论或用户动态压缩为结构化的标签（Tags）或摘要，解决冷启动时的语义缺失问题。

### 核心逻辑

1. 利用 LLM 的 Prompt Engineering 提取关键属性（Category, Style, Scene, Target Audience）。
2. 将提取的标签转化为 One-hot 或 Multi-hot 特征喂给下游模型。

### 实践价值

极大地缓解了冷启动问题。对于没有交互记录的新商品，LLM 提取的“隐式属性”能让模型迅速将其归类到相似的特征空间。

**参考案例：**
[https://arxiv.org/abs/2205.08084](https://arxiv.org/abs/2205.08084)

---

## **方案 B：固定编码器（Fixed Encoder）**

直接调用 LLM（如 BERT、Llama 的隐藏层）获取 Item 或 User 的向量，输入到下游的传统排序模型中。

### 核心逻辑

1. **Item Side:**
   输入商品标题、描述，输出 Item Embedding。

2. **User Side:**
   将用户历史行为序列（标题序列）输入 LLM，输出 User Embedding。

3. **融合：**
   这些向量与传统的 ID Embedding 拼接（Concat），进入后续的交叉层（Cross Network）。

---

## **进阶算法：对齐微调（Contrastive Learning / Alignment）**

> **目标：** 将 LLM 学到的语义向量与推荐系统中的协同过滤（CF）向量对齐，使 LLM 的“语义相似”与用户真实行为中的“偏好相似”一致。

LLM 生成的 embedding 反映的是 **语言语义相似度**，而推荐系统需要的是 **行为相似度**（点击、购买、转化）。

如果直接用 LLM embedding 做召回或排序，会偏离真实用户偏好。
因此必须让 LLM embedding **向协同过滤空间对齐**。

---

### 方法

用真实交互数据构造正负样本，让 LLM 表示遵循用户行为结构。

---

### 最终得到

一个新的 embedding 空间：

> **既保留 LLM 的语义理解能力，又符合真实用户行为分布**

可用于：

* 向量召回（ANN）
* MMoE / Ranker 输入
* 冷启动
* 多模态推荐


| 维度      | 路线 1：LLM → Token → Embedding | 路线 2：LLM → Embedding |
| ------- | ---------------------------- | -------------------- |
| LLM 角色  | 语义解析器 / 特征工程器                | 表示学习模型               |
| 输出      | 离散 token / tag / field       | 连续向量                 |
| 进入 MMoE | 和普通类别特征一样进 embedding 层       | 直接作为 dense 特征        |
| 是否需要对齐  | ❌ 不需要                        | ✅ 必须                 |
| 训练范式    | 完全由 MMoE 学                   | LLM embedding 先天带偏   |
| 冷启动     | 很强                           | 极强                   |
| 稳定性     | 非常高                          | 依赖对齐质量               |


## 二、 LLM 作为精排与重排器 (LLM as Ranker / Re-ranker)

#### **核心逻辑**

1. **输入：** 用户的历史轨迹 + **候选池 (Candidate Set)**。这个候选池通常由 MMoE 或协同过滤等传统模型预先选出（例如 50-100 个 PID）。
2. **处理：** LLM 逐一分析候选池中的物品属性，并与用户偏好进行点对点匹配。
3. **输出：** 候选池物品的**重新排序顺序**。

在这一阶段，LLM 介入推荐的核心链路。通常用于候选集较小的场景，或者作为传统召回后的精化步骤。

* **方案 A：零样本/少样本排序 (Zero-shot / Few-shot Ranking)**
**做法：** 构造 Prompt，将候选列表送入 LLM，要求其按相关性重新排序。
* *优点：* 无需训练，解释性强。
* *引用：* [Is ChatGPT a Good Recommender? (arXiv)](https://arxiv.org/abs/2304.10149)


* **方案 B：指令微调排序 (Instruction Fine-tuning)**
**做法：** 使用 RankNet 或 ListNet 的逻辑，通过 LoRA 等技术微调 LLM，使其学会根据点击历史输出正确的排序顺序。
* *引用：* [LLM as Explainable Re-Ranker for RS (arXiv 2025)](https://arxiv.org/abs/2512.03439)



---

## 三、 LLM 端到端生成式推荐 (Generative Recommendation)

#### **核心逻辑**

1. **输入：** 仅提供用户的历史轨迹和上下文，**不提供候选池**。
2. **处理：** LLM 将推荐看作是文本生成任务的延续（Next-item Prediction）。
3. **输出：** 直接输出物品的 **Token (ID)**。

这是最前沿的方向，彻底抛弃传统的“评分”模式，将推荐视为**序列生成任务**。

* **方案 A：基于 Prompt 的直接生成 (Prompt-based Generation)**
**做法：** 仅通过 Context 描述，让 LLM “预测”下一个用户可能感兴趣的 ID。
* **方案 B：ID 索引微调 (ID-Indexing & Fine-tuning)**
**做法：** 为每个产品分配唯一的 Token ID（或语义 ID），通过微调 LLM 的输出层，使其直接生成 Item ID。
* *实践：* GenRec 框架。
* *引用：* [GenRec: LLM-Driven Generative Recommendation](https://www.emergentmind.com/articles/2307.00457)


* **方案 C：检索增强生成 (RAG for RS)**
**做法：** 结合向量数据库检索相似用户行为，将其作为上下文输入 LLM，生成最终推荐。
* *引用：* [Retrieval-Augmented Recommender Systems (大数跨境)](https://www.google.com/search?q=https://www.10100.com/article/42804927)



---

#### 参考

* https://www.ymshici.com/tech/2124.html?utm_source=chatgpt.com

* https://www.cnblogs.com/Jcloud/p/19153055

* https://eugeneyan.com/writing/recsys-llm/

* *A Survey on LLMs for Recommendation (Section 3.1)*
  [https://aclanthology.org/2024.lrec-main.886/](https://aclanthology.org/2024.lrec-main.886/)

* *Representation Learning with LLMs (Springer)*
  [https://link.springer.com/article/10.1007/s44163-025-00334-5](https://link.springer.com/article/10.1007/s44163-025-00334-5)


