---
title: 旋转矩阵与惯量张量
date: 2025-09-19
categories: [经典力学, 刚体动力学]
tags: [Euler方程, 旋转矩阵, 惯量张量, 坐标变换]
math: true
---

## 一. 旋转矩阵基础与推导

### 1.1 旋转矩阵的定义与物理意义

设旋转矩阵为：
$$
\mathbf{R}(t) = \exp([\mathbf{n}]_\times \theta(t))
$$

其中：
- $\mathbf{n}$：单位旋转轴向量（$\|\mathbf{n}\| = 1$，$3\times1$）。说明：旋转轴的方向向量，如 $[0,0,1]$ 就是绕 $z$ 轴。
- $\theta(t)$：关于时间的旋转角度（弧度）。说明：旋转多少度（弧度），如 $\pi/2$ 就是 90 度。
- $[\mathbf{n}]_\times$ 是 $\mathbf{n}$ 的反对称（叉乘）矩阵：
  $$
  [\mathbf{n}]_\times =
  \begin{pmatrix}
  0 & -n_z & n_y \\
  n_z & 0 & -n_x \\
  -n_y & n_x & 0
  \end{pmatrix}
  $$

**物理意义**：$\exp([\mathbf{n}]_\times \theta)$ 描述“绕 $\mathbf{n}$ 轴旋转 $\theta$ 角度”后的旋转矩阵。该公式可由 Rodrigues 公式推导得到，后面会证明。

---

### 1.2 旋转矩阵的导数与角速度

#### 1.2.1 旋转矩阵的时间导数

对 $\mathbf{R}(t)$ 求导：
$$
\frac{d\mathbf{R}}{dt} = \dot{\theta} [\mathbf{n}]_\times \mathbf{R}(t)
$$

若定义角速度向量 $\boldsymbol{\omega} = \dot{\theta} \mathbf{n}$，则：
$$
\frac{d\mathbf{R}}{dt} = [\boldsymbol{\omega}]_\times \mathbf{R}(t)
$$

其中：
$$
[\boldsymbol{\omega}]_\times = 
\begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
$$

**简要解释**：如果旋转轴方向不变，旋转角度随时间变化，旋转矩阵的变化率等于“当前角速度对应的叉乘矩阵”乘以当前旋转矩阵。

#### 1.2.2 旋转矩阵导数与角速度反对称矩阵的关系

由旋转矩阵的正交性：
$$
\mathbf{R}\mathbf{R}^\mathrm{T} = \mathbf{I}
$$

对其关于时间求导，得到：
$$
\frac{d}{dt}(\mathbf{R}\mathbf{R}^\mathrm{T}) = \frac{d\mathbf{R}}{dt}\mathbf{R}^\mathrm{T} + \mathbf{R}\frac{d\mathbf{R}^\mathrm{T}}{dt} = 0
$$

注意 $\frac{d\mathbf{R}^\mathrm{T}}{dt} = \left(\frac{d\mathbf{R}}{dt}\right)^\mathrm{T}$，因此：
$$
\frac{d\mathbf{R}}{dt}\mathbf{R}^\mathrm{T} + \left(\frac{d\mathbf{R}}{dt}\mathbf{R}^\mathrm{T}\right)^\mathrm{T} = 0
$$

这说明 $\frac{d\mathbf{R}}{dt}\mathbf{R}^\mathrm{T}$ 是一个反对称矩阵（即 $\mathbf{A}^\mathrm{T} = -\mathbf{A}$）。

而实际上，该矩阵就是角速度的反对称矩阵：
$$
[\boldsymbol{\omega}]_\times =
\begin{pmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{pmatrix}
$$

从而：
$$
\frac{d\mathbf{R}}{dt}\mathbf{R}^\mathrm{T} = [\boldsymbol{\omega}]_\times
$$
即：
$$
\frac{d\mathbf{R}}{dt} = [\boldsymbol{\omega}]_\times \mathbf{R}
$$

**简要解释**：旋转矩阵的变化率和自身的乘积，总是一个“反对称矩阵”，这个矩阵的“向量版本”就是空间中的角速度向量 $\boldsymbol{\omega}$。

---

### 1.3 旋转矩阵的标准表达与常见误区

- 标准写法：$\mathbf{R}(t) = \exp([\mathbf{n}]_\times \theta)$
- 不能写成 $\exp(\mathbf{n} \times \theta)$ 或 $\exp(\mathbf{n} \cdot \theta)$，因为 $\mathbf{n} \times \theta$ 没有定义，$\mathbf{n}\theta$ 只是缩放
- 必须用 $[\mathbf{n}]_\times$，即“$\mathbf{n}$ 的叉乘矩阵”

---

## 二. 惯量张量与坐标变换

### 2.1 坐标系区分与惯量张量变换

- $\mathbf{J}_0$：本体坐标系（Body frame）下的惯量张量，常为对角阵
- $\mathbf{J}$：空间坐标系（World frame）下的惯量张量，随刚体转动而变化

变换公式：
$$
\mathbf{J} = \mathbf{R} \mathbf{J}_0 \mathbf{R}^\mathrm{T}
$$

**简要解释**：飞机自己看自己的转动惯量是固定的，世界看它时，由于姿态变了，转动惯量的方向也跟着变。

#### 2.1.1 为什么不是简单的向量变换？

- 惯量张量是二阶张量，必须用 $\mathbf{J} = \mathbf{R} \mathbf{J}_0 \mathbf{R}^\mathrm{T}$
- 用错变换会导致物理和数学错误

---

### 2.2 对惯量张量求导

$$
\begin{align*}
\frac{d\mathbf{J}}{dt}
&= \frac{d}{dt}\left( \mathbf{R} \mathbf{J}_0 \mathbf{R}^\mathrm{T} \right) \\
&= \dot{\mathbf{R}}\, \mathbf{J}_0\, \mathbf{R}^\mathrm{T} + \mathbf{R}\, \mathbf{J}_0\, \frac{d}{dt}\left(\mathbf{R}^\mathrm{T}\right) \\
&= [\boldsymbol{\omega}]_\times\, \mathbf{R}\, \mathbf{J}_0\, \mathbf{R}^\mathrm{T} + \mathbf{R}\, \mathbf{J}_0\, \left( -\mathbf{R}^\mathrm{T} [\boldsymbol{\omega}]_\times \right) \\
&= [\boldsymbol{\omega}]_\times\, \mathbf{J} - \mathbf{J} [\boldsymbol{\omega}]_\times
\end{align*}
$$

其中 $[\boldsymbol{\omega}]_\times$ 是角速度反对称矩阵。

**简要解释**：惯量张量在空间系下的变化速度，就是角速度叉乘矩阵和惯量张量的前后夹乘之差。

---

## 三. 四旋翼动力学建模与惯量张量

### 3.1 刚体角动量变化率公式

- 定义：Body 系下各主轴的转动惯量惯量张量 $\mathbf{J}$，Body 系下角速度向量 $\mathbf{w}$，角速度反对称矩阵 $[\boldsymbol{\omega}]_\times$
- 利用乘积法则（莱布尼茨法则）对 $\mathbf{J}\mathbf{w}$ 求导，得到：
  $$
  \frac{d}{dt}(\mathbf{J}\mathbf{w}) = \frac{d\mathbf{J}}{dt}\mathbf{w} + \mathbf{J}\frac{d\mathbf{w}}{dt}
  $$
- 在空间坐标系下，惯量张量 $\mathbf{J}$ 会随时间变化。其导数为
  $$
  \frac{d\mathbf{J}}{dt}\mathbf{w} = ([\mathbf{\omega}]_\times \mathbf{J} - \mathbf{J} [\mathbf{\omega}]_\times)\mathbf{w}
  $$
    - 因为$$[\mathbf{\omega}]_\times \mathbf{J}\mathbf{w} = \mathbf{\omega} \times (\mathbf{J}\mathbf{w})$$，所以前一项：$$[\mathbf{\omega}]_\times \mathbf{J} \mathbf{w} = \mathbf{\omega} \times (\mathbf{J}\mathbf{w})$$
    - 因为$$[\mathbf{\omega}]_\times \mathbf{w} =  \mathbf{w}\times \mathbf{w} = 0$$，所以后一项：$$\mathbf{J}[\mathbf{\omega}]_\times  \mathbf{w} = 0$$
  
- 于是有：

  $$
  \frac{d}{dt}(\mathbf{J}\mathbf{w}) =\mathbf{J}\frac{d\mathbf{w}}{dt}+ \boldsymbol{\omega} \times (\mathbf{J}\mathbf{w})
  $$

- 也写作：

  $$
  \begin{cases}
  M_1 = I_1 \dot{\omega}_1 + (I_3 - I_2)\omega_2\omega_3 \\
  M_2 = I_2 \dot{\omega}_2 + (I_1 - I_3)\omega_3\omega_1 \\
  M_3 = I_3 \dot{\omega}_3 + (I_2 - I_1)\omega_1\omega_2 \\
  \end{cases}
  $$

---

### 3.2 四旋翼欧拉方程及陀螺项

在 b 系（Body frame）下，四旋翼动力学的**欧拉方程**依然是：
$$
\mathbf{M} = \mathbf{J} \frac{d\mathbf{w}}{dt} + \mathbf{w} \times (\mathbf{J}\mathbf{w})
$$

- **陀螺项** $\mathbf{w} \times (\mathbf{J}\mathbf{w})$ 不能省略！它反映了角速度分量间的耦合（陀螺效应），在飞控和高动态飞行中非常重要
- 飞机在自己坐标系里转动，惯性耦合依然在起作用，不能只看加速度那一项，否则控制会不准，特别是做大幅度快速机动时
- 只有在角速度极小或单轴旋转时，陀螺项才可以近似忽略

---

## 四. 从旋转矩阵到 Rodrigues 公式 [附]

### 4.1 向量旋转公式

对于任意向量 $\mathbf{v}$ 绕单位向量 $\mathbf{u}$ 旋转角度 $\theta$ 后的新向量（这个公式基于数形结合得到）：

$$
\mathbf{v}_{\text{rot}} = \mathbf{v} \cos\theta + (\mathbf{u} \times \mathbf{v}) \sin\theta + \mathbf{u} (\mathbf{u} \cdot \mathbf{v})(1 - \cos\theta)
$$

考虑到：
- 叉乘可写为：$$\mathbf{u} \times \mathbf{v} = [\mathbf{u}]_\times \mathbf{v}$$，其中$$[\mathbf{u}]_\times$$是$$\mathbf{u}$$的反对称矩阵；
  
- 内积的乘法可写为：$$\mathbf{u} (\mathbf{u} \cdot \mathbf{v}) = (\mathbf{u} \mathbf{u}^\mathrm{T}) \mathbf{v}$$，其中$$\mathbf{u} \mathbf{u}^\mathrm{T}$$是$$\mathbf{u}$$的外积矩阵。

所以有：

$$
\begin{align*}
\mathbf{v}_{\text{rot}}
&= \mathbf{v} \cos\theta + (\mathbf{u} \times \mathbf{v}) \sin\theta + \mathbf{u} (\mathbf{u} \cdot \mathbf{v})(1 - \cos\theta) \\
&= \left[ I \cos\theta + [\mathbf{u}]_\times \sin\theta + \mathbf{u} \mathbf{u}^\mathrm{T}(1 - \cos\theta) \right] \mathbf{v} \\
&= \left[ I + [\mathbf{u}]_\times \sin\theta + [\mathbf{u}]_\times^2 (1 - \cos\theta) \right] \mathbf{v}
\end{align*}
$$

又由于$$ \mathbf{\mathbf{v}_{\text{rot}}}  = \mathbf{R} \mathbf{v}$$

所以问题是$$\mathbf{R} = I + [\mathbf{u}]_\times \sin\theta + [\mathbf{u}]_\times^2 (1 - \cos\theta)$$吗？下一节证明。

### 4.2 从矩阵指数到 Rodrigues 公式的推导

已知旋转矩阵可表示为矩阵指数：
$$
\mathbf{R} = \exp([\mathbf{u}]_\times \theta)
$$

其中反对称矩阵 $[\mathbf{u}]_\times$ 定义为：
$$
[\mathbf{u}]_\times = 
\begin{pmatrix}
0 & -u_3 & u_2 \\
u_3 & 0 & -u_1 \\
-u_2 & u_1 & 0
\end{pmatrix}
$$

矩阵指数定义为泰勒级数：
$$
\exp(A) = \sum_{k=0}^{\infty} \frac{A^k}{k!}
$$
其中 $A = [\mathbf{u}]_\times \theta$。

$$
\exp([\mathbf{u}]_\times \theta) = I + ([\mathbf{u}]_\times \theta) + \frac{([\mathbf{u}]_\times \theta)^2}{2!} + \frac{([\mathbf{u}]_\times \theta)^3}{3!} + \cdots
$$

利用反对称矩阵的循环性质：$$[\mathbf{u}]_\times^3 = -[\mathbf{u}]_\times$$，$$[\mathbf{u}]_\times^4 = -[\mathbf{u}]_\times^2$$，$$[\mathbf{u}]_\times^5 = [\mathbf{u}]_\times$$

将级数按奇偶次幂分组：

- **奇次幂项**：$$\theta[\mathbf{u}]_\times - \frac{\theta^3}{3!}[\mathbf{u}]_\times + \frac{\theta^5}{5!}[\mathbf{u}]_\times - \cdots = \sin\theta \cdot [\mathbf{u}]_\times$$
- **偶次幂项**：$$\frac{\theta^2}{2!}[\mathbf{u}]_\times^2 - \frac{\theta^4}{4!}[\mathbf{u}]_\times^2 + \cdots = (1 - \cos\theta) \cdot [\mathbf{u}]_\times^2$$

最终得到简洁形式：

$$
\mathbf{R} = \exp([\mathbf{u}]_\times \theta) = I + [\mathbf{u}]_\times \sin\theta + [\mathbf{u}]_\times^2 (1 - \cos\theta)
$$

### 4.3 几何与代数意义

**物理本质**：

- 矩阵指数 $\exp([\mathbf{u}]_\times \theta)$ 表示将无穷小旋转累积为有限旋转
- 反对称矩阵 $[\mathbf{u}]_\times$ 对应旋转的瞬时角速度

**数学内涵**：

- $SO(3)$ 李群（旋转矩阵）与 $\mathfrak{so}(3)$ 李代数（反对称矩阵）通过指数映射连接
- Rodrigues 公式是李群-李代数对应关系的显式实现

### 4.4 旋转矩阵的几种表达形式

用旋转轴和度数表示：
$$
\mathbf{R} =
\begin{pmatrix}
\cos\theta + u_1^2 (1 - \cos\theta) & u_1 u_2 (1 - \cos\theta) - u_3 \sin\theta & u_1 u_3 (1 - \cos\theta) + u_2 \sin\theta \\
u_2 u_1 (1 - \cos\theta) + u_3 \sin\theta & \cos\theta + u_2^2 (1 - \cos\theta) & u_2 u_3 (1 - \cos\theta) - u_1 \sin\theta \\
u_3 u_1 (1 - \cos\theta) - u_2 \sin\theta & u_3 u_2 (1 - \cos\theta) + u_1 \sin\theta & \cos\theta + u_3^2 (1 - \cos\theta)
\end{pmatrix}
$$

用欧拉角表示：$\mathbf{R}_{\text{Euler}} = R_z(\psi) R_y(\theta) R_x(\phi)$

$$
\mathbf{R}_{\text{Euler}} =
\begin{pmatrix}
\cos\psi & -\sin\psi & 0 \\
\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{pmatrix}
\begin{pmatrix}
1 & 0 & 0 \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi & \cos\phi
\end{pmatrix}
$$

$$
\mathbf{R}_{\text{Euler}} =
\begin{pmatrix}
\cos\psi \cos\theta & \cos\psi \sin\theta \sin\phi - \sin\psi \cos\phi & \cos\psi \sin\theta \cos\phi + \sin\psi \sin\phi \\
\sin\psi \cos\theta & \sin\psi \sin\theta \sin\phi + \cos\psi \cos\phi & \sin\psi \sin\theta \cos\phi - \cos\psi \sin\phi \\
-\sin\theta & \cos\theta \sin\phi & \cos\theta \cos\phi
\end{pmatrix}
$$

已知飞机在机体系下的受力$ \mathbf{T}_b $，根据旋转矩阵计算NED系下的受力$ \mathbf{T}_e $ ：

$$
\mathbf{T}_e =  R_z(\psi) R_y(\theta) R_x(\phi)\mathbf{T}_b
$$
