---
title: 协同过滤与ALS矩阵分解学习
date: 2025-09-13
categories: [推荐系统, 算法原理]
tags: [协同过滤, ALS, 隐语义, Python, FAISS]
math: true
---

## 一. 前言

推荐系统已经成为互联网产品提升用户体验和业务增长的核心技术。从电商、短视频到社交网络，推荐算法无处不在。本文以“协同过滤”为主线，结合实际代码，学习推荐系统的经典流程、核心算法原理、工程实现与常见问题。

---

## 二. 推荐系统的典型流程

### 2.1 主要阶段

一个成熟的推荐系统通常包含以下几个关键阶段：

1. **召回（Recall）**  
   - 目的：从海量物品中快速筛选出一小部分“候选物品”，这些物品大概率是用户喜欢的。
   - 常用方法：
     - 协同过滤（Collaborative Filtering, CF）
     - 基于内容的召回（Content-based）
     - 热门物品、最新物品
     - 召回模型（如Embedding召回、图召回等）

2. **粗排（Pre-ranking）**  
   - 目的：对召回阶段得到的候选物品进行初步排序，进一步缩小范围。
   - 常用方法：
     - 简单打分模型（如逻辑回归LR、GBDT等）
     - 简单特征（如用户兴趣、物品热度）

3. **精排（Ranking）**  
   - 目的：对粗排后的物品进行精细排序，最终选出Top N推荐给用户。
   - 常用方法：
     - 复杂排序模型（如DNN、Wide&Deep等）
     - 综合更多特征（用户画像、上下文、实时行为等）

4. **重排序/过滤（Post-processing）**  
   - 目的：去重、过滤掉不合适的内容，保证推荐结果的多样性、新颖性等。

---

## 三. 协同过滤基础与原理

### 3.1 协同过滤简介

协同过滤是一种基于用户历史行为数据（如收藏、点击、评分等），通过挖掘用户兴趣和物品特性，实现个性化推荐的方法。它是推荐系统中最经典、应用最广的算法之一，尤其常用于召回阶段。

### 3.2 两大类协同过滤

1. **基于用户的协同过滤（User-based CF）**  

   - 找到与目标用户兴趣相似的其他用户（邻居），推荐这些用户喜欢但目标用户没接触过的物品。
   
   - 余弦相似度公式：
     
     $$
     \text{sim}(u, v) = \frac{|I_u \cap I_v|}{\sqrt{|I_u| \cdot |I_v|}}
     $$
     
     推荐得分：
     
     $$
     \text{score}(u, i) = \sum_{v \in N(u)} \text{sim}(u, v) \cdot r_{v,i}
     $$

2. **基于物品的协同过滤（Item-based CF）**  

   - 找到与目标用户喜欢的物品相似的其他物品，推荐这些相似物品。
   
   - 余弦相似度公式：
     
     $$
     \text{sim}(i, j) = \frac{|U_i \cap U_j|}{\sqrt{|U_i| \cdot |U_j|}}
     $$
     
     推荐得分：
     
     $$
     \text{score}(u, j) = \sum_{i \in I_u} \text{sim}(i, j)
     $$

### 3.3 协同过滤的实现方式

- 邻域方法：直接计算用户或物品之间的相似度，基于相似度做推荐。
- 矩阵分解方法：将用户-物品行为矩阵分解为低维向量（如SVD、ALS），用向量之间的内积或余弦相似度做推荐。

### 3.4 优缺点分析

| 优点                           | 缺点                                    |
| ------------------------------ | --------------------------------------- |
| 不依赖物品内容，能发现潜在兴趣 | 冷启动问题（新用户/新物品）、稀疏性问题 |

---

## 四. ALS矩阵分解与隐语义模型

### 4.1 ALS的基本思想与目标函数

ALS（Alternating Least Squares，交替最小二乘法）是一种经典的协同过滤矩阵分解算法。它假设每个用户和物品都可以用一个低维隐向量表示，用户对物品的兴趣可以用这两个向量的内积衡量。

**目标函数**：

$$
\min_{P, Q} \sum_{u,i} (R_{ui} - p_u^T q_i)^2 + \lambda (\|p_u\|^2 + \|q_i\|^2)
$$

- $R_{ui}$：用户 $u$ 对物品 $i$ 的行为（如1表示收藏，0表示未收藏）
- $p_u$：用户 $u$ 的隐向量
- $q_i$：物品 $i$ 的隐向量
- $\lambda$：正则化参数

### 4.2 ALS的优化过程与公式推导

ALS采用交替优化策略，每次固定一组变量（用户或物品隐向量），优化另一组变量。

**单步优化公式（以优化用户隐向量为例）**：

$$
p_u = \left( \sum_{i \in I_u} q_i q_i^T + \lambda I \right)^{-1} \left( \sum_{i \in I_u} R_{ui} q_i \right)
$$

- $\sum_{i \in I_u} q_i q_i^T$ 是一个 $k \times k$ 矩阵（$k$ 是隐向量维度）
- $\sum_{i \in I_u} R_{ui} q_i$ 是一个 $k$ 维向量

> **Note**: 这是一个标准的带L2正则的最小二乘问题，每步都能解析求解。

**详细推导过程**：

对 $p_u$ 求导并令导数为0，得到：

$$
\frac{\partial L}{\partial p_u} = -2 \sum_{i \in I_u} (R_{ui} - p_u^T q_i) q_i + 2\lambda p_u = 0
$$

整理后：

$$
\sum_{i \in I_u} R_{ui} q_i = \left( \sum_{i \in I_u} q_i q_i^T + \lambda I \right) p_u
$$

解得：

$$
p_u = \left( \sum_{i \in I_u} q_i q_i^T + \lambda I \right)^{-1} \left( \sum_{i \in I_u} R_{ui} q_i \right)
$$

### 4.3 ALS的特性与收敛性

- 每步优化为凸问题，损失函数单调下降。
- 损失函数有下界，最终收敛到稳定状态。
- ALS属于坐标下降法的特例，每一步都不会让损失增加，只会减少或不变，最终一定会收敛。

> **Note**: ALS只能保证收敛到一个局部最优解，但实际推荐系统中已能很好地反映用户和物品的潜在兴趣和属性。

### 4.4 ALS隐向量的意义与相似性

- 用户隐向量：代表用户的潜在兴趣特征（如喜欢甜品、景点等，自动学出来）。
- 物品隐向量：代表物品的潜在属性特征（如属于甜品类、景点类等，同样自动学出来）。

**为什么“相似”的会“相近”？**

- 如果两个用户和很多相同的物品发生了交互，ALS会让他们的隐向量变得相似。
- 如果两个物品被很多相同的用户喜欢，ALS会让它们的隐向量变得相似。
- 推荐时只需计算用户和物品隐向量的内积。

### 4.5 ALS用户向量与物品向量平均的关系辨析

- **用户收藏物品向量平均法**：直接将用户已收藏物品的Embedding取平均，作为用户兴趣向量：

  $$
  \mathbf{p}_u^{\text{avg}} = \frac{1}{|I_u|} \sum_{i \in I_u} \mathbf{q}_i
  $$

- **ALS用户向量**：通过矩阵分解优化得到，实际是对收藏物品向量的“加权平均+正则化+全局优化”：

  $$
  \mathbf{p}_u^{\text{ALS}} = \left( \sum_{i \in I_u} \mathbf{q}_i \mathbf{q}_i^T + \lambda I \right)^{-1} \left( \sum_{i \in I_u} R_{ui} \mathbf{q}_i \right)
  $$

- **关系辨析**：
    - 平均法仅聚合用户历史物品，忽略物品间协同和全局分布。
    - ALS则在聚合基础上引入加权、正则化和全局优化，使用户向量更能拟合实际兴趣。
    - 当无正则化且物品向量正交时，ALS结果接近平均。
- **总结**：ALS用户向量本质上是收藏物品向量的“加权平均+正则化+全局优化”结果，比简单平均更精准表达用户兴趣。

---

## 五. 实践代码与工程实现

### 5.1 数据构造与交互矩阵

```python
import numpy as np
from scipy.sparse import lil_matrix

n_users = 100
n_pois = 10000
np.random.seed(2024)

# 每个用户收藏 50~100 个 POI
user_poi = []
for i in range(n_users):
    num = np.random.randint(50, 101)
    pois = np.random.choice(n_pois, num, replace=False)
    user_poi.append(pois)

# 构建用户-POI稀疏交互矩阵
R = lil_matrix((n_users, n_pois), dtype=np.float32)
for uid, pois in enumerate(user_poi):
    R[uid, pois] = 1.0
```

### 5.2 ALS矩阵分解与隐向量获取

```python
import implicit

R_coo = R.tocoo()
R_item_user = R_coo.transpose().tocsr()  # implicit要求item-user格式

model = implicit.als.AlternatingLeastSquares(factors=32, regularization=0.1, iterations=10)
model.fit(R_item_user)

user_vecs = model.user_factors  # shape: (n_users, factors)
poi_vecs = model.item_factors   # shape: (n_pois, factors)
```

### 5.3 FAISS高效向量召回

```python
import faiss

d = poi_vecs.shape[1]
index = faiss.IndexFlatIP(d)  # 内积相似度
faiss.normalize_L2(poi_vecs)  # 归一化为单位向量
index.add(poi_vecs.astype(np.float32))
```

### 5.4 推荐流程实现

```python
def recall_pois_for_user(user_id, topk=50):
    user_vec = user_vecs[user_id].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(user_vec)
    D, I = index.search(user_vec, topk + 50)  # 多召回一些，后面去重
    already = set(user_poi[user_id])
    recs = [i for i in I[0] if i not in already]
    return recs[:topk]

recall_list = recall_pois_for_user(0, topk=50)
print("用户0召回POI:", recall_list)
```

### 5.5 常用库说明

- **implicit**：高效的隐语义推荐库，支持ALS、BPR等算法，适合大规模稀疏数据。
- **faiss**：Facebook AI Similarity Search，高效的向量相似度检索库，支持CPU/GPU加速。
- **scipy.sparse**：用于存储和操作大规模稀疏矩阵。

### 5.6 该系统中ALS的应用方式

**当前方案属于“隐语义模型下的物品召回”，本质上更接近于“基于物品的协同过滤”思想。**

- 推荐流程是：拿用户隐向量，找相似的物品隐向量，推荐这些物品。
- 物品的隐向量是通过所有用户的行为学习出来的，反映了物品之间的协同关系。

**作用和意义：**

- 压缩信息：把用户和物品都变成低维向量，便于存储和计算。
- 发现潜在关系：隐向量可以捕捉到用户和物品之间的潜在兴趣和特征。
- 高效推荐：只需要计算用户和物品向量的内积，就能给出推荐分数，效率高，扩展性强。

---

## 六. 常见问题与思考

### 6.1 ALS与协同过滤的关系

- ALS属于协同过滤的矩阵分解方法，通过只依赖用户-物品行为数据，自动从数据中学习出用户兴趣和物品特征。
- ALS分解后的隐向量空间，相似的用户/物品会自然靠近，这也是后续用向量检索（如FAISS）做召回的理论基础。

### 6.2 ALS的优化过程

- 交替最小二乘法每次优化一个变量（用户或物品隐向量），都能直接解析求解，效率远高于SGD等方法。
- 这种交替优化保证每一步损失都不会增加，最终一定收敛到一个局部最优。

### 6.3 推荐系统整体流程

- 推荐系统分为召回、粗排、精排等阶段，每个阶段用不同的模型和特征，既保证效率又提升精度。
- 协同过滤，尤其是ALS矩阵分解，常用于召回阶段，快速从海量物品中筛选高相关候选集。

### 6.4 协同过滤的局限与改进方向

- 冷启动和稀疏性问题是协同过滤的天然短板，实际工程中常结合内容特征、图召回等多种方法。
- 深度学习、图神经网络等新方法正在不断提升推荐系统的能力。

---
