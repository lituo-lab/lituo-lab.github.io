---
title: 模型预测控制（MPC）简介
date: 2025-07-01
categories: [控制理论, 优化控制]
tags: [MPC, Automatic]
math: true
---

# 模型预测控制（MPC）简介

## 什么是模型预测控制？

模型预测控制（Model Predictive Control, MPC）是一种先进的过程控制方法，广泛应用于工业自动化、机器人、汽车控制和能源管理等领域。MPC通过使用系统的动态模型来预测未来一段时间内的系统行为，并基于优化算法计算最优控制输入。

## MPC的基本原理

MPC的核心思想可以概括为以下三个步骤：

1. **预测**：利用系统模型预测未来一段时间（预测时域）的系统状态
2. **优化**：求解一个有限时域的最优控制问题，最小化目标函数
3. **滚动实施**：只实施当前时刻的最优控制输入，下一时刻重复整个过程

这种"滚动时域"策略使MPC能够处理系统约束并适应模型误差。

## MPC的数学表达

典型的MPC问题可以表示如下：

$$
\begin{aligned}
\min_{u} & \sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N \\
\text{s.t.} & \quad x_{k+1} = f(x_k, u_k) \\
& \quad x_k \in \mathcal{X}, u_k \in \mathcal{U}
\end{aligned}
$$

其中：
- $x_k$ 是系统状态

- $u_k$是控制输入
- $Q$, $R$, $P$是权重矩阵
- $\mathcal{X}$和$\mathcal{U}$是状态和输入的约束集

## MPC的优势

1. **显式处理约束**：可以直接考虑状态和输入的约束
2. **多变量控制**：天然适合多输入多输出系统
3. **前馈补偿**：可以处理可测干扰
4. **灵活性**：目标函数和约束可以灵活调整

## MPC的应用领域

- 化工过程控制
- 自动驾驶车辆
- 航空航天
- 能源管理系统
- 机器人控制

## MPC的挑战

1. **计算复杂度**：需要在线求解优化问题
2. **模型精度**：性能依赖于模型的准确性
3. **实时性要求**：需要在采样时间内完成计算

## 总结

MPC是一种强大的控制策略，特别适合处理多变量约束系统。随着计算能力的提升和优化算法的发展，MPC的应用范围正在不断扩大。

## 延伸阅读

- [Model Predictive Control System Design and Implementation Using MATLAB®](https://www.springer.com/gp/book/9781848823310)
- [Predictive Control: With Constraints](https://www.pearson.com/us/higher-education/program/Maciejowski-Predictive-Control-with-Constraints/PGM228000.html)
