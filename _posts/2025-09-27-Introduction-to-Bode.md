---
title: 闭环系统稳定性与伯德图分析
date: 2025-09-27
categories: [控制理论, 系统分析]
tags: [频域分析, 稳定性, 相位裕度, 伯德图, 时滞系统]
math: true
---

## 一、引言：从物理直觉到控制系统

### 1.1 共振现象的启示

控制理论的学习往往始于对物理现象的深入思考。一个经典问题是：为什么当外力频率接近弹簧系统的**固有频率** $\omega_n$ 时，会发生共振导致幅值急剧增大？

在探索过程中，一个关键发现是：在伯德图上，位移 $X(s)$ 对外力 $F(s)$ 的响应在共振峰处对应的相位**并非 $0^\circ$（同相），而是 $-90^\circ$（滞后）**。

### 1.2 能量视角的突破

**关键洞察**：共振的本质在于能量的持续注入。功率 $P = F \cdot v$（力与速度的乘积）决定了系统能量的变化。由于速度 $\dot{x}$ 比位移 $x$ **超前 $90^\circ$**，当位移滞后力 $90^\circ$ 时，速度恰好与力**同相（$0^\circ$）**，这正是外力对系统做正功、持续注入能量的理想条件。

这一发现再次说明了将物理直觉、时域行为和频域分析相结合的重要性。为此，我们将在MATLAB中构建主动控制系统来"驯服"弹簧共振。

---

## 二、时域实验：主动阻尼器的双面性

### 2.1 系统建模与控制目标

考虑质量-弹簧-阻尼系统：

$$
m\ddot{x} + c\dot{x} + k_p x = F_c
$$

采用**速度负反馈**控制律，在物理上对应理想的阻尼器：

$$
F_c(t) = -k_c \dot{x}(t)
$$

### 2.2 理想情况：无时滞控制

```matlab
% 理想控制器实现
Fc_ideal = -kc * v_ideal(i-1);
a_ideal = (Fc_ideal - c*v_ideal(i-1) - kp*x_ideal(i-1)) / m;
v_ideal(i) = v_ideal(i-1) + a_ideal * dt;
x_ideal(i) = x_ideal(i-1) + v_ideal(i-1) * dt;
```

设置 $k_c = 5$，系统从初始位移 $x_0=1$ 释放，结果符合预期：系统快速收敛，振动被有效抑制。

### 2.3 现实挑战：引入时间延迟

实际系统中，传感、计算和执行都需要时间。引入 $0.4$ 秒的时间延迟：

$$
F_c(t) = -k_c \dot{x}(t - \tau)
$$

```matlab
% 时滞控制器实现
delay_steps = round(tau/dt);
if i > delay_steps
    Fc_delay = -kc * v_delay(i-delay_steps);
else
    Fc_delay = -kc * v_delay(1);  % 初始阶段处理
end
```

### 2.4 失稳现象与分析

当 $k_c=5, \tau=0.4\text{ s}$ 时，系统出现**剧烈发散**。

> **物理机制分析**：时滞导致控制力 $F_c$ 基于**历史速度** $\dot{x}(t-\tau)$。当质量块向左运动（$\dot{x}$ 为负）时，控制器可能基于之前向右运动（$\dot{x}$ 为正）的采样值施加向左的力，导致**力与速度同向**，本应消耗能量的阻尼力转变为**做正功**的助推力。

---

## 三、频域分析：伯德图与稳定性判据

### 3.1 开环传递函数构建

开环传递函数 $L(s)$ 为：

$$
L(s) = C(s) \cdot G(s) = \left( k_c s e^{-\tau s} \right) \cdot \left( \frac{1}{ms^2 + cs + k_p} \right)
$$

### 3.2 频域分析代码

```matlab
% 系统参数 
m = 1; c = 0.1; kp = 1; kc = 5; tau = 0.4;

% 开环传递函数
s = tf('s');
G_plant = 1 / (m*s^2 + c*s + kp);
L_ideal = G_plant * (kc * s);
L_delay = G_plant * (kc * s * exp(-tau*s));

% 3. 绘制伯德图
figure; bodeplot(L_ideal, L_delay); grid on;
legend('无时滞 L_{ideal}', ['有时滞 L_{delay}']);

% 稳定性裕度计算
[Gm_ideal, Pm_ideal, Wcg_ideal, Wcp_ideal] = margin(L_ideal);
[Gm_delay, Pm_delay, Wcg_delay, Wcp_delay] = margin(L_delay);
```

### 3.3 稳定性裕度的物理意义

- **相位裕度 (Phase Margin, PM)**：在**增益穿越频率** $\omega_{gc}$ 处，系统相位距离 $-180^\circ$ 的余量，反映对**时间延迟**的容忍度
- **增益裕度 (Gain Margin, GM)**：在**相位穿越频率** $\omega_{pc}$ 处，系统增益距离 1（0 dB）的余量，反映对**增益变化**的容忍度

### 3.4 关键结果：负相位裕度

| 系统条件                     | 相位裕度      | 稳定性结论 |
| ---------------------------- | ------------- | ---------- |
| 无时滞 ($\tau=0$)            | $89.5^\circ$  | **稳定**   |
| 有时滞 ($\tau=0.4\text{ s}$) | $-21.3^\circ$ | **不稳定** |

时滞引入的相位滞后 $-\omega\tau$ 显著降低相位裕度，**PM < 0** 从理论上解释了时域的失稳现象。

---

## 四、深入探索：参数影响与频率特性

### 4.1 增益 $k_c$ 对稳定性的影响

**现象**：较小 $\tau$ 时，$k_c=1$ 系统收敛，$k_c=10$ 系统发散。

**频域解释**：
- 增大 $k_c$ → 幅频曲线上移
- 增益穿越频率 $\omega_{gc}$ 右移（增大）
- 时滞相位滞后 $-\omega\tau$ 随频率增大而增强
- 相位裕度被进一步压缩，最终导致失稳

### 4.2 发散频率的本质

测量与计算结果对比：

| 频率类型                   | 数值 (Hz) |
| -------------------------- | --------- |
| 系统固有频率 $f_n$         | 0.16      |
| 时域观测发散频率 $f_{osc}$ | 0.685     |
| 增益穿越频率 $f_{gc}$      | 0.637     |
| 相位穿越频率 $f_{pc}$      | 0.826     |

系统发散频率**并非**固有频率，而是由闭环系统的不稳定极点决定，实际位于 $f_{gc}$ 和 $f_{pc}$ 之间：$0.637 < 0.685 < 0.826$。

---

## 五、理论推导：闭环极点与发散频率

### 5.1 闭环系统建模

带时滞速度反馈的系统方程为：

$$
m\ddot{x} + c\dot{x} + k_p x = -k_c \dot{x}(t - \tau)
$$

将控制力代入系统方程，得到：

$$
ms^2 X(s) + cs X(s) + k_p X(s) + k_c s e^{-\tau s} X(s) = F_{ext}(s)
$$

闭环传递函数（外部扰动 $F_{ext}(s)$ 为输入，$X(s)$ 为输出）为：

$$
T(s) = \frac{X(s)}{F_{ext}(s)} = \frac{1}{ms^2 + cs + k_p + k_c s e^{-\tau s}}
$$

特征方程：

$$
ms^2 + cs + k_p + k_c s e^{-\tau s} = 0
$$

### 5.2 相位裕度与稳定性的数学联系

相位裕度 $\text{PM} = 180^\circ + \angle L(j\omega_{gc})$。当 $\text{PM} < 0$ 时：
- 奈奎斯特曲线包围 $(-1, 0)$ 点
- 闭环系统存在实部为正的极点 $s = \sigma + j\omega_{osc}$ ($\sigma > 0$)
- 时域响应呈指数发散 $e^{\sigma t}$，震荡频率为 $\omega_{osc}$ 

### 5.3 数值求解：Pade近似方法

```matlab
% 系统参数 
m = 1; c = 0.1; kp = 1; kc = 5; tau = 0.4;

% 构造闭环系统 T(s)
s = tf('s');
G_loop = (kc * s) / (m*s^2 + c*s + kp);
C_delay = exp(-tau*s);
T_delay = feedback(G_loop * C_delay, 1);

% Pade近似求解闭环极点
N_pade = 10;
T_delay_approximated = pade(T_delay, N_pade);
poles = pole(T_delay_approximated);
unstable_poles = poles(real(poles) > 0);

% 提取主导极点
sigma = real(unstable_poles(1));
omega_osc = abs(imag(unstable_poles(1)));
f_osc = omega_osc / (2*pi);
```

### 5.4 理论验证

| 指标       | 理论值             | 观测值             | 结论     |
| ---------- | ------------------ | ------------------ | -------- |
| 不稳定极点 | $0.4862 + j4.3002$ | N/A                | 确认发散 |
| 发散频率   | $0.6844\text{ Hz}$ | $0.6854\text{ Hz}$ | 高度吻合 |

---

## 六、MATLAB代码实现

### 6.1 时域仿真代码

```matlab
clear; close all; clc;

% 物理参数
m = 1;      % 质量 (kg)
c = 0.1;    % 阻尼系数 (N*s/m)
kp = 1;     % 固有弹簧刚度 (N/m)

kc = 5;     % 速度反馈增益 (N/m)
tau = 0.4;  % 时间延迟 (s)

% 仿真时间
dt = 0.001;  T_total = 30; 
t = 0:dt:T_total; 
N = length(t);
delay_steps = round(tau/dt);

% 初始化
x_ideal = zeros(1, N); v_ideal = zeros(1, N); x_ideal(1) = 1;
x_delay = zeros(1, N); v_delay = zeros(1, N); x_delay(1) = 1;


% 主循环
for i = 2:N
    % 无时滞控制器
    Fc_ideal = -kc * v_ideal(i-1);
    a_ideal = (Fc_ideal - c*v_ideal(i-1) - kp*x_ideal(i-1)) / m;
    v_ideal(i) = v_ideal(i-1) + a_ideal * dt;
    x_ideal(i) = x_ideal(i-1) + v_ideal(i-1) * dt;
    
    % 有时滞控制器
    if i > delay_steps
        Fc_delay = -kc * v_delay(i-delay_steps);
    else
        Fc_delay = -kc * v_delay(1);  % 初始时滞期间用初值
    end
    a_delay = (Fc_delay - c*v_delay(i-1) - kp*x_delay(i-1)) / m;
    v_delay(i) = v_delay(i-1) + a_delay * dt;
    x_delay(i) = x_delay(i-1) + v_delay(i-1) * dt;
end

% 绘图
figure;
plot(t, x_ideal, 'b', 'LineWidth', 2); hold on;
plot(t, x_delay, 'r-.', 'LineWidth', 2);
legend('无时滞控制器', '有时滞控制器');
xlabel('时间 (s)');
ylabel('位移 (m)');
title('质量-弹簧系统时域仿真（有无时滞控制器）');
grid on;
```

### 6.2 频域分析代码

```matlab
clc; close all; clear

% 1. 定义系统参数和控制参数
m = 1;      % 质量 (kg)
c = 0.1;    % 阻尼系数 (N*s/m)
kp = 1;     % 固有弹簧刚度 (N/m)

kc = 5;     % 速度反馈增益 (N/m)
tau = 0.4;  % 时间延迟 (s)

% 2. 构造开环传递函数模型 L(s)
s = tf('s'); % 拉普拉斯变量

% 物理系统的传递函数 G_plant(s) = X(s)/F_ext(s)
G_plant = 1 / (m*s^2 + c*s + kp);

% 速度反馈控制器 C_v(s) = Kc * s (无延迟部分)
C_v_ideal = kc * s;

% 2.1. 无时滞开环系统 L_ideal(s)
L_ideal = G_plant * (kc * s);

% 2.2. 有时滞开环系统 L_delay(s)
L_delay = G_plant * (kc * s * exp(-tau*s));

% 3. 绘制伯德图并进行裕度分析
figure('Name','速度反馈开环系统伯德图对比');
h = bodeplot(L_ideal, L_delay);

setoptions(h, 'MagUnits', 'dB', 'PhaseUnits', 'deg', 'Grid', 'on');
title('速度反馈开环系统伯德图对比 (L_{ideal} vs L_{delay})');
legend('无时滞 L_{ideal} (\tau=0)', ['有时滞 L_{delay} (\tau=', num2str(tau), 's)'], 'Location', 'SouthWest');
grid on;

% 4. 稳定性裕度计算
disp('--- 稳定性裕度分析 (速度反馈增益 kc = 5) ---');

% 4.1. 无时滞系统分析
[Gm_ideal, Pm_ideal, Wcg_ideal, Wcp_ideal] = margin(L_ideal);
disp('【无时滞系统 L_ideal】:');
disp(['  相位裕度 (PM): ', num2str(Pm_ideal), ' 度']);
disp(['  增益裕度 (GM): ', num2str(20*log10(Gm_ideal)), ' dB']);
disp(['  结论: 闭环系统稳定 (PM>0)，与仿真结果一致。']);

% 4.2. 有时滞系统分析
[Gm_delay, Pm_delay, Wcg_delay, Wcp_delay] = margin(L_delay);
disp(['【有时滞系统 L_delay (\tau=', num2str(tau), 's)】:']);
disp(['  相位裕度 (PM): ', num2str(Pm_delay), ' 度']);
disp(['  增益裕度 (GM): ', num2str(20*log10(Gm_delay)), ' dB']);
disp(['  结论: 闭环系统不稳定 (PM < 0)，与您的仿真结果一致。']);
```

### 6.3 Nyquist稳定性分析

```matlab
 %% 系统参数定义
clc; close all; clear

m = 1; c = 0.1; kp = 1; kc = 5; tau = 0.4;
s = tf('s');
G_plant = 1 / (m*s^2 + c*s + kp);
L_delay = G_plant * (kc * s * exp(-tau*s));

%% 1. 绘制 Bode 图（含时滞系统）
figure('Name', 'Bode Diagram', 'NumberTitle', 'off');
bode(L_delay, {1e-2, 1e2});  % 频率范围从 0.01 到 100 rad/s
grid on;
title('Bode Diagram of L_{delay}(s) with Time Delay \tau = 0.4s');

%% 2. 绘制奈奎斯特曲线（Nyquist Plot）
figure('Name', 'Nyquist Plot', 'NumberTitle', 'off');
nyquist(L_delay, {1e-2, 1e2});  % 指定频率范围
grid on;
title('Nyquist Plot of L_{delay}(s) with Time Delay \tau = 0.4s');

% 可选：添加单位圆和临界点 (-1, 0) 以评估稳定性
hold on;
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--');  % 单位圆
plot(-1, 0, 'ro', 'MarkerFaceColor', 'r');  % 临界点 (-1, 0)
text(-1.1, 0.1, '(-1, 0)', 'Color', 'red', 'FontSize', 10);
xlabel('Real Axis');
ylabel('Imaginary Axis');
legend('Nyquist Curve', 'Unit Circle', 'Critical Point (-1,0)', 'Location', 'best');
hold off;
```
---

## 七、结论与启示

通过本次从物理现象到理论分析的完整探索，我们获得以下重要认识：

1. **物理直觉为基础**：能量视角（力与速度的相位关系）是理解共振和失稳现象的关键
2. **时域-频域互补**：时域仿真揭示现象，频域分析诊断病因，两者结合提供完整理解
3. **时滞的破坏性**：时间延迟通过相位滞后 $-\omega\tau$ 侵蚀相位裕度，是控制系统失稳的主要因素
4. **裕度的工程意义**：相位裕度和增益裕度分别量化系统对时滞和增益变化的容错能力
5. **理论闭环验证**：从现象观测到开环分析，再到闭环极点推导，完成完整的理论验证链条

这一研究过程体现了控制工程中理论分析、数值仿真和物理直觉相结合的方法论价值，为复杂系统控制设计提供了重要借鉴。