# 📝 MMoE 多任务模型训练总结笔记

## 1️⃣ 模型架构：MMoE (Multi-gate Mixture-of-Experts)

**核心思想**：

* 共享 **专家网络（Experts）**，通过 **任务专属门控（Gates）** 为不同任务选择不同的专家组合。
* 支持多任务训练（分类 + 回归），并使用 **融合权重（Fusion α）** 对最终得分进行加权组合。

### 1.1 输入

* **数值特征**：`x_num`，直接传入
* **类别特征**：`x_cat`，通过 Embedding 层映射到向量
* **Embedding自动维度**：`auto_emb_dim = 8` (可以根据类别基数动态设置)

### 1.2 模块组成

| 模块                                | 说明                                                       |
| --------------------------------- | -------------------------------------------------------- |
| **Embeddings**                    | 为每个类别特征生成 embedding，最后与数值特征拼接形成输入向量                      |
| **Experts**                       | 多个全连接网络，每个 `expert_hidden=128`，ReLU 激活 + dropout         |
| **Gates**                         | 每个任务对应 gate（`gate_cls`, `gate_reg`），输出每个专家的权重，使用 softmax |
| **Task Towers**                   | 任务专属网络：分类和回归，各自 2 层全连接 + dropout                         |
| **Fusion α**                      | 可学习参数，通过 sigmoid 将分类和回归结果加权融合                            |
| **不确定性加权（Uncertainty Weighting）** | 对分类和回归损失引入自适应权重 (`log_sigma_cls`, `log_sigma_reg`)       |

### 1.3 前向传播逻辑

1. Embedding + 数值特征拼接
2. 输入每个专家网络 → 得到专家输出矩阵 `[batch, hidden, num_experts]`
3. Gate 计算专家权重 → 对专家输出加权求和 → 任务特定的混合表示
4. 进入任务塔 → 输出分类 `cls_out` 和回归 `reg_out`
5. Sigmoid 标准化 → 通过 Fusion α 生成最终 `final_score`

---

## 2️⃣ 损失函数设计

### 2.1 分类损失

* **BCEWithLogitsLoss**，可调 `pos_weight` 处理类别不平衡

### 2.2 回归损失

* **MSE Loss**，仅对正样本（如借款用户）计算

### 2.3 排序损失 (Pairwise Ranking Loss)

* 对同一用户的正样本进行两两排序约束
* 使用 `pairwise_hinge_loss`：
  [
  L_\text{rank} = \frac{1}{N} \sum_{i,j} \max(0, \text{margin} - (s_i - s_j)) \quad \text{if } t_i > t_j
  ]

### 2.4 多任务自动加权

* 利用不确定性权重：
  [
  \text{Loss} = e^{-\sigma_\text{cls}} L_\text{cls} + \sigma_\text{cls} + e^{-\sigma_\text{reg}} L_\text{reg} + \sigma_\text{reg} + \lambda_\text{rank} L_\text{rank}
  ]

---

## 3️⃣ 数据处理与 Dataset

### 3.1 类别特征编码

* `SimpleCategoryEncoder`：自动生成 1-based 编码，0 作为未知值
* 获取每列的基数（用于 embedding 层）

### 3.2 数值特征标准化

* 使用 `StandardScaler`
* **回归目标** `loan_amount` 先截断 `30000` 后 `log1p` 处理（防止极端值）

### 3.3 Dataset 封装

* `MultiTaskDataset` 支持训练/验证/测试
* 可返回 `uid` 用于排序损失计算

---

## 4️⃣ 训练流程

### 4.1 时间窗口切分

* 滑动窗口：

  * 训练集：`train_days`
  * 验证集：`val_days`
  * 测试集：`test_days`
  * 步长：`window_step`

### 4.2 模型训练循环

1. **训练阶段**：

   * 数据搬运到设备
   * 前向传播得到 `cls_pred`, `reg_pred`, `final_score`
   * 计算多任务损失 + 排序损失
   * 反向传播与优化
2. **验证阶段**：

   * 推理得到最终分数
   * 计算业务相关性（实际借款金额与 final_score 的相关系数）
3. **早停策略**：

   * 基于验证集业务相关性 (`val_corr`)
   * `patience` 控制未改善轮数

---

## 5️⃣ 监控指标

| 类别         | 指标                                             |
| ---------- | ---------------------------------------------- |
| **损失与权重**  | train_loss, fusion α, σ_cls, σ_reg             |
| **任务表现**   | train/val AUC, train/val RMSE                  |
| **专家使用情况** | gate_cls_mean, gate_reg_mean                   |
| **核心早停指标** | 验证集业务相关性 val_metric (final_score 与实际借款金额的相关系数) |

### 5.1 可视化

* Loss + α 曲线
* 分类 AUC
* 回归 RMSE
* 专家使用率
* 业务相关性趋势

---

## 6️⃣ 推理与评估

* `run_inference`：

  * 输出：分类预测、回归预测、最终融合分数、真实标签
  * 收集门控平均权重用于分析专家分配
* `calculate_metrics`：

  * 分类：AUC
  * 回归（正样本）：RMSE, MAE
  * 综合融合指标：final_score 与实际借款金额相关性

---

## 7️⃣ 特点与优化点

1. **多任务共享 + 专家选择** → 模型灵活，提升不同任务表现
2. **不确定性自动加权** → 动态平衡分类/回归任务
3. **融合预测分数** → final_score 用于业务相关性优化
4. **用户内排序损失** → 对正样本的排序优化，提升推荐/借款排名质量
5. **门控分析** → 可视化专家使用情况，辅助模型调优

---

## 8️⃣ 总结流程图 (文字版)

```
输入：数值特征 + 类别特征
           │
         Embedding
           │
      拼接 → 输入到Experts
           │
      Experts输出矩阵
           │
       Gate加权 → 任务特定混合表示
       ┌─────────┴─────────┐
   分类塔                   回归塔
     │                        │
   cls_out                  reg_out
     │                        │
 Sigmoid标准化            Sigmoid标准化
     │                        │
     └───── Fusion α ────────┘
                │
          final_score
```

