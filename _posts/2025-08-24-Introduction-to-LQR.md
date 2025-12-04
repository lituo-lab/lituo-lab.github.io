---
title: 倒立摆仿真与LQR控制实现
date: 2025-08-24
categories: [控制理论, 仿真实现]
tags: [LQR, 倒立摆, 控制算法, Python, 仿真]
math: true
---

## 一. 物理建模与系统分析

### 1.1 数学物理模型

<img src="/assets/img/2025-08-24-Introduction-to-LQR.png" alt="倒立摆模型" style="zoom:40%;" />

### 1.2 拉格朗日方程推导

- **系统动能**：
  $$
  T = \underbrace{\frac{1}{2}M\dot{x}^2}_{\text{小车动能}} + \underbrace{\frac{1}{2}m\left(\dot{x}^2 + l^2\dot{\theta}^2 + 2l\dot{x}\dot{\theta}\cos\theta\right) + \frac{1}{2}\frac{1}{12}m(2l)^2\dot{\theta}^2}_{\text{杆的动能}}
  $$

- **系统势能**：
  $$
  V = mgl\cos\theta
  $$

- **拉格朗日函数**：
  $$
  \mathcal{L} = T - V = \frac{1}{2}(M + m)\dot{x}^2 + \frac{2}{3}ml^2\dot{\theta}^2 + ml\dot{x}\dot{\theta}\cos\theta - mgl\cos\theta
  $$

### 1.3 运动方程推导

- **关于 $x$ 的欧拉-拉格朗日方程**：
  $$
  \frac{d}{dt}\left(\frac{\partial\mathcal{L}}{\partial\dot{x}}\right) - \frac{\partial\mathcal{L}}{\partial x} = F \\ 
  \Rightarrow (M + m)\ddot{x} + ml\ddot{\theta}\cos\theta - ml\dot{\theta}^2\sin\theta = F
  $$

- **关于 $\theta$ 的欧拉-拉格朗日方程**：
  $$
  \frac{d}{dt}\left(\frac{\partial\mathcal{L}}{\partial\dot{\theta}}\right) - \frac{\partial\mathcal{L}}{\partial\theta} = 0 \\ 
  \Rightarrow \frac{4}{3}ml^2\ddot{\theta} + ml\ddot{x}\cos\theta - mgl\sin\theta = 0
  $$

### 1.4 状态空间表达式构建

- **矩阵形式表达**：
  $$
  \begin{bmatrix} 
  M + m & m l \cos\theta \\ 
  m l \cos\theta & \frac{4}{3} m l^2 
  \end{bmatrix} 
  \begin{bmatrix} 
  \ddot{x} \\ 
  \ddot{\theta} 
  \end{bmatrix} 
  = 
  \begin{bmatrix} 
  u + m l \dot{\theta}^2 \sin\theta \\ 
  m g l \sin\theta 
  \end{bmatrix}
  $$

- **加速度求解表达式**：
  $$
  \Delta = (M + m) \cdot \frac{4}{3} m l^2 - (m l \cos\theta)^2 
  $$
  
$$
  \begin{bmatrix} 
  \ddot{x} \\ 
  \ddot{\theta} 
  \end{bmatrix} = 
  \frac{1}{\Delta} 
  \begin{bmatrix} 
  \frac{4}{3} m l^2 & -m l \cos\theta \\ 
  - m l \cos\theta & M + m 
  \end{bmatrix} 
  \begin{bmatrix} 
  u + m l \dot{\theta}^2 \sin\theta \\ 
  m g l \sin\theta 
  \end{bmatrix}
$$

- **状态空间形式**：
  定义状态变量 $x_1 = x$, $x_2 = \dot{x}$, $x_3 = \theta$, $x_4 = \dot{\theta}$，则
  $$
  \begin{cases}
  \dot{x}_1 = x_2 \\
  \dot{x}_2 = \displaystyle \frac{ \frac{4}{3} m l^2 (u + m l x_4^2 \sin x_3) - m^2 l^2 g \sin x_3 \cos x_3 }{ (M + m) \cdot \frac{4}{3} m l^2 - (m l \cos x_3)^2 } \\
  \dot{x}_3 = x_4 \\
  \dot{x}_4 = \displaystyle \frac{ -(m l \cos x_3)(u + m l x_4^2 \sin x_3) + (M + m) m g l \sin x_3 }{ (M + m) \cdot \frac{4}{3} m l^2 - (m l \cos x_3)^2 }
  \end{cases}
  $$


### 1.5 仿真结果与可视化

![倒立摆动画演示](/assets/img/2025-08-24-Introduction-to-LQR.gif)

> **Note**: 仿真采用四阶龙格-库塔法进行数值积分，确保计算精度和稳定性。


## 二. LQR控制器设计与线性化

### 2.1 线性化状态空间表达式

- **小角度近似**：
  $$
  \sin\theta \approx \theta,\quad \cos\theta \approx 1,\quad \dot{\theta}^2 \sin\theta \approx 0
  $$
  $$
  \begin{bmatrix}
  M + m & m l \\
  m l & \frac{4}{3} m l^2
  \end{bmatrix}
  \begin{bmatrix}
  \ddot{x} \\
  \ddot{\theta}
  \end{bmatrix}
  =
  \begin{bmatrix}
  u \\
  m g l \theta
  \end{bmatrix}
  $$

- **近似解**：
  $$
  \ddot{x} = \frac{4 u - 3 m g x_3}{4M + m}
  $$
  $$
  \ddot{\theta} = \frac{ -3 u + 3 (M + m) g x_3 }{ l (4M + m) }
  $$

- **状态空间表达式**：
  $$
  \dot{\mathbf{x}} = A \mathbf{x} + B u,\quad \mathbf{x} = \begin{bmatrix} x \\ \dot{x} \\ \theta \\ \dot{\theta} \end{bmatrix}
  $$
  $$
  A = \begin{bmatrix}
  0 & 1 & 0 & 0 \\
  0 & 0 & -\frac{3 m g}{4M + m} & 0 \\
  0 & 0 & 0 & 1 \\
  0 & 0 & \frac{3 (M + m) g}{l (4M + m)} & 0
  \end{bmatrix},\quad
  B = \begin{bmatrix}
  0 \\
  \frac{4}{4M + m} \\
  0 \\
  -\frac{3}{l (4M + m)}
  \end{bmatrix}
  $$

### 2.2 LQR控制理论基础

- **性能指标**：
  $$
  J = \int_{0}^{\infty} \left[ (x - x_{target})^T Q (x - x_{target}) + u^T R u \right] dt
  $$

- **最优控制律**：
  $$
  u^* = -K(x - x_{target})
  $$
  其中$K$通过求解Riccati方程得到：
  $$
  A^T P + P A - P B R^{-1} B^T P + Q = 0
  $$
  $$
  K = R^{-1} B^T P
  $$

## 三. Python实现与仿真分析

### 3.1 系统建模与实现

```python
class InvertedPendulum:
    def __init__(self, params, u_max=10.0):
        self.M = params['M']
        self.m = params['m']
        self.l = params['l']
        self.g = params['g']
        self.u_max = u_max

    def dynamics(self, x, u):
        u = np.clip(u, -self.u_max, self.u_max)
        x1, x2, x3, x4 = x
        s = np.sin(x3)
        c = np.cos(x3)
        D = (self.M + self.m) * (4/3 * self.m * self.l**2) - (self.m * self.l * c)**2
        f2 = ((4/3 * self.m * self.l**2) * (u + self.m * self.l * x4**2 * s) - 
              self.m**2 * self.l**2 * self.g * s * c) / D
        f4 = ((self.M + self.m) * self.m * self.g * self.l * s - 
              self.m * self.l * c * (u + self.m * self.l * x4**2 * s)) / D
        return np.array([x2, f2, x4, f4])
```

### 3.2 LQR控制器实现

```python
class LQRController():
    def __init__(self, system, Q, R, x_target=None):
        self.system = system
        self.Q = Q
        self.R = R
        self.x_target = x_target if x_target is not None else np.zeros(4)
        A, B = self.linearize_dynamics()
        P = solve_continuous_are(A, B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ B.T @ P

    def get_control(self, x, t):
        error = self.x_target - x
        u = self.K @ error
        return u.item()
```

## 四. 完整代码实现

```python
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class InvertedPendulum:
    def __init__(self, params, u_max=10.0):
        self.M = params['M']
        self.m = params['m']
        self.l = params['l']
        self.g = params['g']
        self.u_max = u_max

    def dynamics(self, x, u):
        u = np.clip(u, -self.u_max, self.u_max)
        x1, x2, x3, x4 = x
        s = np.sin(x3)
        c = np.cos(x3)
        D = (self.M + self.m) * (4/3 * self.m * self.l**2) - (self.m * self.l * c)**2
        f2 = ((4/3 * self.m * self.l**2) * (u + self.m * self.l * x4**2 * s) - 
              self.m**2 * self.l**2 * self.g * s * c) / D
        f4 = ((self.M + self.m) * self.m * self.g * self.l * s - 
              self.m * self.l * c * (u + self.m * self.l * x4**2 * s)) / D
        return np.array([x2, f2, x4, f4])
        
    def rk4_step(self, x, u, dt):
        f = self.dynamics
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class LQRController():
    def __init__(self, system, Q, R, x_target=None):
        self.system = system
        self.Q = Q
        self.R = R
        self.x_target = x_target if x_target is not None else np.zeros(4)
        A, B = self.linearize_dynamics()
        P = solve_continuous_are(A, B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ B.T @ P

    def linearize_dynamics(self):
        M = self.system.M
        m = self.system.m
        l = self.system.l
        g = self.system.g
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, (-m * g) / M, 0],
            [0, 0, 0, 1],
            [0, 0, (M + m) * g / (M * l), 0]
        ])
        B = np.array([
            [0],
            [1 / M],
            [0],
            [-1 / (M * l)]
        ])
        return A, B

    def get_control(self, x, t):
        error = self.x_target - x
        u = self.K @ error
        return u.item()

class Simulator:
    def __init__(self, system, controller, x0, dt=0.01, T=10.0, control_interval=1, noise_std=None):
        self.system = system
        self.controller = controller
        self.dt = dt
        self.T = T
        self.N = int(T / dt)
        self.control_interval = control_interval
        self.x = np.array(x0)
        self.history = []
        self.u_history = []
        self.t_history = []
        if noise_std is None:
            self.noise_std = np.array([0.0, 0.0, 0.02, 0.05])
        else:
            self.noise_std = np.array(noise_std)

    def run(self):
        u = 0.0
        for i in range(self.N):
            t = i * self.dt
            measured_x = self.x + np.random.randn(4) * self.noise_std
            if i % self.control_interval == 0:
                u = self.controller.get_control(measured_x, t)
            self.x = self.system.rk4_step(self.x, u, self.dt)
            self.history.append(self.x.copy())
            self.u_history.append(np.clip(u, -self.system.u_max, self.system.u_max))
            self.t_history.append(t)
        self.history = np.array(self.history)
        self.u_history = np.array(self.u_history)
        self.t_history = np.array(self.t_history)
    
    def plot_and_animate(self, save_path='lqr_pendulum.gif'):

        fig = plt.figure(figsize=(12, 4), dpi=100)
        gs = fig.add_gridspec(2, 4)
        ax_anim = fig.add_subplot(gs[:, :2])
        ax1 = fig.add_subplot(gs[0, 2:])
        ax2 = fig.add_subplot(gs[1, 2:])
        
        pendulum_length = self.system.l
        cart_width = 0.3
        cart_height = 0.15
        
        ax_anim.set_xlim(-2, 2)
        ax_anim.set_ylim(-0.2, 1.2)
        ax_anim.set_title("Inverted Pendulum Animation (LQR)")
        ax_anim.set_xlabel('x (m)')
        ax_anim.set_ylabel('y (m)')
        
        cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='b')
        pendulum_line, = ax_anim.plot([], [], 'r-', lw=3)
        pendulum_mass, = ax_anim.plot([], [], 'ro', ms=12)
        ax_anim.add_patch(cart_patch)
        time_text = ax_anim.text(0.05, 0.90, '', transform=ax_anim.transAxes, fontsize=14, color='k')
        
        ax1.plot(self.t_history, self.history[:, 0], label='Cart x')
        ax1.plot(self.t_history, self.history[:, 2], label='Theta')
        dot1, = ax1.plot([self.t_history[0]], [self.history[0, 0]], 'ko', ms=8)
        dot2, = ax1.plot([self.t_history[0]], [self.history[0, 2]], 'mo', ms=8)
        ax1.set_ylabel('x (m) / theta (rad)')
        ax1.legend()
        ax1.grid()
        

        ax2.plot(self.t_history, self.u_history, label='Force u')
        dot3, = ax2.plot([self.t_history[0]], [self.u_history[0]], 'ko', ms=8)
        ax2.set_ylabel('Force u (N)')
        ax2.set_xlabel('Time (s)')
        ax2.legend()
        ax2.grid()
        
        plt.tight_layout()

        def init():
            cart_patch.set_xy((0 - cart_width / 2, 0))
            pendulum_line.set_data([], [])
            pendulum_mass.set_data([], [])
            time_text.set_text('')
            dot1.set_data([self.t_history[0]], [self.history[0, 0]])
            dot2.set_data([self.t_history[0]], [self.history[0, 2]])
            dot3.set_data([self.t_history[0]], [self.u_history[0]])
            return cart_patch, pendulum_line, pendulum_mass, time_text, dot1, dot2, dot3
        
        def update(i):
            x = self.history[i, 0]
            theta = self.history[i, 2]
            cart_patch.set_xy((x - cart_width / 2, 0))
            px = x + pendulum_length * np.sin(theta)
            py = cart_height / 2 + pendulum_length * np.cos(theta)
            pendulum_line.set_data([x, px], [cart_height / 2, py])
            pendulum_mass.set_data([px], [py])
            time_text.set_text(f'Time = {self.t_history[i]:.2f} s')
            dot1.set_data([self.t_history[i]], [self.history[i, 0]])
            dot2.set_data([self.t_history[i]], [self.history[i, 2]])
            if i < len(self.u_history):
                dot3.set_data([self.t_history[i]], [self.u_history[i]])
            return cart_patch, pendulum_line, pendulum_mass, time_text, dot1, dot2, dot3

        dt_ms = self.dt * 1000
        fps = int(1/self.dt)
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.history), init_func=init,
            blit=True, interval=dt_ms)
        
        ani.save(save_path, writer='pillow', fps=fps)
        plt.close(fig)

if __name__ == "__main__":
    params = {'M': 1.0, 'm': 0.3, 'l': 0.5, 'g': 9.81}
    x0 = [1.0, 2.0, 0.1, 0.3]
    x_target = [0.0, 0.0, 0.0, 0.0]

    system = InvertedPendulum(params, u_max=10)
    Q = np.diag([1, 1, 10, 1])
    R = np.array([[1]])

    controller = LQRController(system, Q, R, x_target=x_target)
    
    sim = Simulator(system, controller, x0, dt=0.02, T=10.0, control_interval=2)
    sim.run()
    sim.plot_and_animate()
```

## 五. 总结

本文完整实现了倒立摆系统的非线性建模、LQR控制器设计与仿真验证。通过拉格朗日方程推导建立了精确的动力学模型，采用小角度近似进行线性化处理，并基于Riccati方程求解得到最优控制律。Python实现包含了完整的仿真框架和可视化功能，为控制理论教学和工程应用提供了有价值的参考。

> **Warning**: 实际应用中需注意控制力饱和约束和系统参数不确定性对控制性能的影响。
