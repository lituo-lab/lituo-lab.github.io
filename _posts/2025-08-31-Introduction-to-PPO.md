---
title: 倒立摆仿真与PPO控制实现
date: 2025-08-31
categories: [强化学习, 控制理论]
tags: [PPO, 倒立摆, Actor-Critic, GAE]
math: true
---

## 一. 学习背景

本次学习以强化学习中的PPO算法和Actor-Critic结构为主线，目标是理解其理论基础，掌握连续动作空间下的代码实现，并能解释每一步的物理意义和工程动机。

### 1.1 物理模型

<img src="/assets/img/Inverted_pendulum.png" alt="倒立摆模型" style="zoom:40%;" />

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
  
### 1.4 控制效果

![倒立摆动画演示](/assets/img/pendulum_ppo.gif)

---


## 二. PPO与Actor-Critic算法原理

### 2.1 PPO简介

- PPO是一种基于策略梯度的强化学习算法，适用于连续和离散动作空间。
- 其核心是通过clip机制限制每次策略更新的幅度，保证训练过程的稳定性和高效性。

### 2.2 Actor-Critic结构与三个子网络

- PPO通常采用Actor-Critic结构，包含三个子网络（实际常合并为一个网络）：
    - **n1：动作均值网络（mean）**：输出动作分布的均值。
    - **n2：动作方差网络（std）**：输出动作分布的标准差，控制策略的探索性。
    - **n3：价值网络（value）**：输出当前状态下未来累计奖励的期望 $V(s)$。
- n1和n2通过PPO损失让"好动作"概率变大，"坏动作"概率变小。
- n3通过均方误差（MSE）损失，拟合每个状态的真实累计回报。

### 2.3 优势函数（Advantage）、Q、V与GAE公式及理解

#### 2.3.1 基本定义

- **Q函数**：在状态 $s_t$ 下采取动作 $a_t$，然后后续按策略行动，未来累计奖励的期望。
- **V函数**：在状态 $s_t$ 下，按照当前策略采样动作，未来累计奖励的平均水平。
- **优势函数**：衡量当前动作比平均水平好多少。
  $$
  A_t = Q(s_t, a_t) - V(s_t)
  $$
- **TD误差**：
  $$
  \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
  $$
  采取动作 $a_t$ 后事情变好了多少？

#### 2.3.2 GAE（Generalized Advantage Estimation）核心思想与公式

- GAE不是只看一步的TD误差，而是把未来所有TD误差都考虑进来，且越远的影响越小（指数衰减）。
- 数学公式：
  $$
  A_t^{\text{GAE}(\gamma, \lambda)} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots
  $$
  其中 $\lambda$（$0 \leq \lambda \leq 1$）控制长期和短期的平衡，$\lambda$ 越大，考虑的步数越多。
- **物理意义**：$r_t$、$V(s_t)$、$V(s_{t+1})$ 都是"累计奖励"量纲，TD误差代表每一步对价值预测的修正。GAE就是把这些修正加权累加，动态调整对未来奖励的估计。
- **大白话**：GAE就是"把以后每一步的预测误差都考虑进来，离得越远影响越小"。这样既能兼顾短期准确性，又能平滑长期趋势，让训练又快又稳。

#### 2.3.3 advantages + values[:-1] 与 Q、V的关系

- GAE优势 $A_t^{\text{GAE}}$ 是 $Q(s_t, a_t) - V(s_t)$ 的近似。
- 在PPO代码中，常用 `returns = advantages + values[:-1]`，即
  $$
  Q(s_t, a_t) \approx A_t^{\text{GAE}} + V(s_t)
  $$
- 训练Critic时，用 $Q(s_t, a_t)$ 的估计作为目标，让 $V(s_t)$ 拟合它，从而逐步逼近真实的"平均水平"。
- **大白话**：
    - $V(s_t)$ 是"我通常能拿多少分"；
    - $A_t^{\text{GAE}}$ 是"我这次比平均多/少拿了多少分"；
    - 两者相加，就是"我这次实际能拿到的分数期望"（$Q(s_t, a_t)$）。
    - $V(s_t)$靠近这次的分数，但是多次训练，$V(s_t)$实际山靠近的是期望分数，即我通常能拿多少分。
- 这样做的好处是，既利用了真实奖励的信息，又用到了当前价值网络的预测，降低了方差，提高了训练效率。

---

## 三. PPO连续动作实现与代码结构

### 3.1 环境与网络设计

- 以倒立摆为例，环境支持连续动作，step接口接收float型控制量。
- 网络结构示例：

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh() )
        
        self.actor_mean = nn.Linear(32, action_dim)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * np.log(action_std_init))
        self.critic = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.shared(x)
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(x)
        return mean, std, value
```

- mean/std决定动作高斯分布，value用于辅助训练。

### 3.2 采样与数据收集流程

- 每个epoch采集一批数据，记录状态、动作、奖励、done、value、logprob：

```python
for step in range(steps_per_epoch):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    mean, std, value = net(state_tensor)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    action_clipped = action.clamp(-env.u_max, env.u_max)
    logprob = dist.log_prob(action).sum(dim=-1).item()
    next_state, reward, done, _ = env.step(action_clipped.detach().numpy()[0])
    # 存储数据...
    state = next_state
    if done:
        state = env.reset()
```

### 3.3 PPO训练主循环与损失函数设计

- 采样数据后，先用GAE算出每一步的优势和目标回报（returns）：

```python
def compute_gae(rewards, values, dones, gamma=0.98, lam=0.95):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t]
        nextvalue = values[t + 1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values[:-1]
    return advantages, returns
```

- 多轮小批量训练，核心损失函数如下：

```python
mean, std, values_pred = net(mb_states)
dist = torch.distributions.Normal(mean, std)
new_logprobs = dist.log_prob(mb_actions).sum(dim=-1)
ratio = torch.exp(new_logprobs - mb_logprobs)
surr1 = ratio * mb_advantages
surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
actor_loss = -torch.min(surr1, surr2).mean()
critic_loss = nn.MSELoss()(values_pred.squeeze(), mb_returns)
entropy = dist.entropy().sum(dim=-1).mean()
loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 3.4 核心代码片段与注释

- **logprob计算**：多维动作的概率乘起来，取对数后相加。
- **clip损失**：如果策略变化太大就"拦一下"，防止训练崩坏。
- **熵项**：鼓励策略分布"发散"，防止策略太快变成"死板的动作"。

---

## 四. 常见疑问与解答

### 4.1 PPO为什么能学到好策略？

- 策略损失鼓励好动作概率越来越大，坏动作概率越来越小。
- clip机制防止策略更新过大，保证训练稳定。
- critic网络降低方差，提高训练效率。
- 熵项鼓励探索，避免陷入局部最优。
- 大白话：PPO的本质就是"好动作多做点，坏动作少做点，每次别改太猛"。

### 4.2 Critic（value）网络的训练与作用

- Critic网络通过MSE损失拟合采样得到的returns（累计奖励）。
- 作用：
    1. 作为优势估计的基线，降低方差。
    2. 辅助actor学习更优策略。
- 训练时必不可少，推理时只用actor即可。
- 大白话：训练时必须有value网络帮忙算"到底好多少"，部署时只用actor选动作就行。

### 4.3 为什么不直接用累计rewards当advantage？

- 直接用累加rewards（Monte Carlo回报）噪声大，训练不稳定、收敛慢。
- value网络作为"基线"，能大幅降低方差，训练更稳。
- GAE再进一步，结合短期和长期信息，效果更好。

### 4.4 GAE/returns为什么用到value函数？会不会死循环？

- GAE和returns用value函数是为了降低方差、加速收敛，这叫"自举"思想。
- 每次采样时用当前网络的value预测"未来"，不是死循环，因为每次采样都包含真实奖励，网络会不断修正自己。

### 4.5 Value网络初始是随机的，为什么能收敛？

- 虽然初始参数是随机的，但每次采样后都能用真实累计奖励做"标签"，用MSE损失不断修正参数。
- 采样-修正-再采样-再修正，最终value网络能准确预测未来累计奖励。
- 和普通神经网络做回归任务没本质区别。
- 大白话：一开始乱猜，见得多了、修正多了，自然就准了。

---

## 五. PPO损失、熵正则与clip机制直观解释

- **clip机制**：PPO损失的clip机制就像"护栏"，保证策略更新不会一步跳太远。
- **熵正则项**：就像是"多鼓励你去试试新花样"，别太快死板下来。
- **多轮小批量训练**：是为了"高效利用数据，不浪费经验"。
- 数学表达：
    - clip损失：$ \text{actor\_loss} = -\mathrm{mean}(\min(surr1, surr2)) $
    - 熵正则：$ \text{loss} = \text{actor\_loss} + 0.5 \times \text{critic\_loss} - 0.01 \times \text{entropy} $

---

## 六. 代码总结

```python
import numpy as np
import os, torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ========== 倒立摆环境 ==========
class PendulumEnv:
    def __init__(self, params, u_max=10.0, dt=0.02, max_steps=500):
        self.system = InvertedPendulum(params, u_max)
        self.u_max = u_max
        self.dt = dt
        self.max_steps = max_steps
        self.state_dim = 4
        self.action_dim = 1
        self.x_target = np.zeros(4)
        self.reset()
        
    def reset(self, x0=None):
        if x0 is not None:
            self.state = np.array(x0)
        else:
            self.state = np.random.uniform(
                low = [-2, -0.2, -np.pi/4, -0.2], 
                high= [ 2,  0.2,  np.pi/4,  0.2])
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        u = float(np.clip(action, -self.u_max, self.u_max).item())
        x = self.state
        x_next = self.system.rk4_step(x, u, self.dt)
        self.state = x_next
        self.steps += 1
        
        done = ( self.steps >= self.max_steps 
                or abs(self.state[2]) > np.pi/2 
                or abs(self.state[0]) > 10.0)
        
        w = np.array([0.1, 0.01, 0.5, 0.01]) * 5
        reward = - np.sum(w * (self.state - self.x_target) ** 2)
        reward += 0.5 * (abs(self.state[0]) < 0.05) * (abs(self.state[2]) < 0.05)
        reward += 2.0 * (abs(self.state[0]) < 0.01) * (abs(self.state[2]) < 0.01)
        reward -= 5.0 * (done and self.steps < self.max_steps)
        
        return self.state.copy(), reward, done, {}

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
        f2 = ((4/3 * self.m * self.l**2) * (u + self.m * self.l * x4**2 * s) - self.m**2 * self.l**2 * self.g * s * c) / D
        f4 = ((self.M + self.m) * self.m * self.g * self.l * s - self.m * self.l * c * (u + self.m * self.l * x4**2 * s)) / D
        return np.array([x2, f2, x4, f4])
    def rk4_step(self, x, u, dt):
        f = self.dynamics
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ========== ActorCritic ==========
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh() )
        
        self.actor_mean = nn.Linear(32, action_dim)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * np.log(action_std_init))
        self.critic = nn.Linear(32, 1)
    def forward(self, x):
        x = self.shared(x)
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(x)
        return mean, std, value

# ========== GAE ==========
def compute_gae(rewards, values, dones, gamma=0.98, lam=0.95):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t]
        nextvalue = values[t + 1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values[:-1]
    return advantages, returns

# ========== PPO训练 ==========
def train_ppo(model_path="ppo_continuous.pth", load_model=False):
    params = {'M': 1.0, 'm': 0.3, 'l': 0.5, 'g': 9.81}
    env = PendulumEnv(params, u_max=10, dt=0.02, max_steps=500)
    state_dim = env.state_dim
    action_dim = env.action_dim
    net = ActorCritic(state_dim, action_dim)
    if load_model and os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded pre-trained model from {model_path}")
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    epochs = 50000
    steps_per_epoch = 500
    minibatch_size  = 64
    update_epochs   = 8
    gamma = 0.98
    lam = 0.95
    clip_eps = 0.2
    rewards_history = []
    for epoch in range(epochs):
        states, actions, rewards, dones, values, logprobs = [], [], [], [], [], []
        state = env.reset()
        for step in range(steps_per_epoch):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std, value = net(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1).item()
            action_np = action.detach().numpy()[0]
            next_state, reward, done, _ = env.step(action_np)
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            logprobs.append(logprob)
            state = next_state
            if done:
                state = env.reset()
        # 最后一个value
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        _, _, last_value = net(state_tensor)
        values.append(last_value.item())
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        # GAE
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 转为Tensor
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        logprobs_tensor = torch.tensor(logprobs, dtype=torch.float32)
        # PPO多轮更新
        inds = np.arange(steps_per_epoch)
        for _ in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, steps_per_epoch, minibatch_size):
                mb_inds = inds[start:start+minibatch_size]
                mb_states = states_tensor[mb_inds]
                mb_actions = actions_tensor[mb_inds]
                mb_returns = returns_tensor[mb_inds]
                mb_advantages = advantages_tensor[mb_inds]
                mb_logprobs = logprobs_tensor[mb_inds]
                mean, std, values_pred = net(mb_states)
                dist = torch.distributions.Normal(mean, std)
                new_logprobs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = torch.exp(new_logprobs - mb_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values_pred.squeeze(), mb_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 评估
        eval_rewards = []
        for _ in range(3):
            state = env.reset()
            ep_reward = 0
            for _ in range(env.max_steps):
                with torch.no_grad():
                    mean, std, _ = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = mean[0].cpu().numpy()
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = next_state
                if done:
                    break
            eval_rewards.append(ep_reward)
        avg_reward = np.mean(eval_rewards)
        rewards_history.append(avg_reward)
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}, avg reward: {avg_reward:.2f}")
            torch.save(net.state_dict(), model_path)
            plt.plot(rewards_history)
            plt.xlabel("Epoch")
            plt.ylabel("Average Test Reward")
            plt.title(f"PPO Training Curve (Continuous Action) - Epoch {epoch+1}")
            plt.tight_layout()
            plt.savefig("ppo_continuous_training_curve.png")
            plt.close()

# ========== PPO测试 ==========
def test_ppo(model_path="ppo_continuous.pth", save_ani=False):
    params = {'M': 1.0, 'm': 0.3, 'l': 0.5, 'g': 9.81}
    env = PendulumEnv(params, u_max=10, dt=0.02, max_steps=500)
    state_dim = env.state_dim
    action_dim = env.action_dim
    net = ActorCritic(state_dim, action_dim)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    
    state = env.reset()
    states = [state.copy()]
    actions = []
    rewards = []
    for _ in range(env.max_steps):
        with torch.no_grad():
            mean, std, _ = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = mean[0].cpu().numpy()
        clipped_action = np.clip(action, -env.u_max, env.u_max)
        next_state, reward, done, _ = env.step(clipped_action)
        actions.append(float(clipped_action.item()))
        rewards.append(reward)
        states.append(next_state.copy())
        state = next_state
        if done:
            break
    
    states = np.array(states)
    actions = np.array(actions)
    t_history = np.arange(len(states)) * env.dt

    # 画轨迹
    plt.figure()
    plt.plot(t_history, states[:, 0], label="Loaction x")
    plt.plot(t_history, states[:, 2], label="Theta")
    plt.legend()
    plt.title("Test Trajectory (Random x0)")
    plt.tight_layout()
    plt.savefig("ppo_continuous_test_trajectory.png")
    plt.close()

    if save_ani:  # 只有在save_ani为True时才保存
        # 保存动图
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(2, 4)
        ax_anim = fig.add_subplot(gs[:, :2])
        ax1 = fig.add_subplot(gs[0, 2:])
        ax2 = fig.add_subplot(gs[1, 2:])

        pendulum_length = env.system.l
        cart_width = 0.3
        cart_height = 0.15
        ax_anim.set_xlim(-2, 2)
        ax_anim.set_ylim(-0.2, 1.2)
        ax_anim.set_title("Inverted Pendulum Animation")
        ax_anim.set_xlabel('x (m)')
        ax_anim.set_ylabel('y (m)')
        cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='b')
        pendulum_line, = ax_anim.plot([], [], 'r-', lw=3)
        pendulum_mass, = ax_anim.plot([], [], 'ro', ms=12)
        ax_anim.add_patch(cart_patch)
        time_text = ax_anim.text(0.05, 0.90, '', transform=ax_anim.transAxes, fontsize=14, color='k')

        ax1.plot(t_history, states[:, 0], label='Cart x')
        ax1.plot(t_history, states[:, 2], label='Theta')
        dot1, = ax1.plot([t_history[0]], [states[0, 0]], 'ko', ms=8)
        dot2, = ax1.plot([t_history[0]], [states[0, 2]], 'mo', ms=8)
        ax1.set_ylabel('x (m) / theta (rad)')
        ax1.legend()
        ax1.grid()

        ax2.plot(t_history[:-1], actions, label='Force u')
        dot3, = ax2.plot([t_history[0]], [actions[0]], 'ko', ms=8)
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
            dot1.set_data([t_history[0]], [states[0, 0]])
            dot2.set_data([t_history[0]], [states[0, 2]])
            dot3.set_data([t_history[0]], [actions[0]])
            return cart_patch, pendulum_line, pendulum_mass, time_text, dot1, dot2, dot3

        def update(i):
            x = states[i, 0]
            theta = states[i, 2]
            cart_patch.set_xy((x - cart_width / 2, 0))
            px = x + pendulum_length * np.sin(theta)
            py = cart_height / 2 + pendulum_length * np.cos(theta)
            pendulum_line.set_data([x, px], [cart_height / 2, py])
            pendulum_mass.set_data([px], [py])
            time_text.set_text(f'Time = {t_history[i]:.2f} s')
            dot1.set_data([t_history[i]], [states[i, 0]])
            dot2.set_data([t_history[i]], [states[i, 2]])
            if i < len(actions):
                dot3.set_data([t_history[i]], [actions[i]])
            return cart_patch, pendulum_line, pendulum_mass, time_text, dot1, dot2, dot3
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(states), init_func=init,
            blit=True, interval=env.dt * 1000
        )
        ani.save("ppo_continuous_test_result.gif", writer='pillow', fps=int(1/env.dt))
        plt.close(fig)
        print("PPO连续动作测试动图已保存为 ppo_continuous_test_result.gif")

if __name__ == "__main__":
    model_path = "ppo_continuous.pth"
    if 0:  # 设置为1时训练，设置为0时测试
        train_ppo(model_path, load_model=True)
    else:
        test_ppo(model_path, save_ani=False) 
```
