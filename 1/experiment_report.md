# CartPole-v1 强化学习实验报告

## 摘要
本实验使用Q-learning算法实现CartPole-v1环境的平衡任务，通过离散化状态空间和epsilon-greedy策略训练智能体。实验结果显示，在适当参数下，智能体能在5000次训练episode内达到平均奖励170+，证明算法有效性。

## 引言
CartPole-v1是经典的强化学习基准任务，要求智能体通过左右推力保持杆子平衡。状态包括小车位置、速度、杆子角度和角速度。奖励为每步1分，episode在杆倒或超出界限时终止。

本实验采用Q-learning，结合状态离散化和神经网络-free的方法，探索参数对收敛的影响。

## 方法
### 算法概述
- **Q-learning**：时序差分学习，更新公式：  
  `Q(s, a) += α * (r + γ * max(Q(s')) - Q(s, a))`
- **状态离散化**：将连续状态映射到离散bins，使用`np.linspace`定义边界。
- **epsilon-greedy**：以epsilon概率随机动作，否则选择最大Q值动作。

### 代码实现
- **类QLearning**：封装初始化、训练、测试方法。
- **训练流程**：每1000 episode测试10次，记录平均奖励。
- **终止条件**：步数>200、位置>2.4、角度>12°。

## 实验设置
- **环境**：Gymnasium CartPole-v1。
- **参数**：
  - bins: (15, 15, 15, 15)
  - gamma: 0.99
  - alpha: 0.1
  - epsilon: 0.01
  - num_episodes: 8000
- **硬件**：标准PC，Python 3.12，依赖：gymnasium, numpy, matplotlib。

## 结果
- **训练曲线**：奖励随episode增加波动上升，4000 episode后稳定在较高水平。
- **性能指标**：5000 episode时平均奖励约170+，表明策略收敛。
- **可视化**：生成training_plot.png显示奖励vs episode。

## 讨论
- **优势**：简单有效，无需深度网络，适合离散任务。
- **局限**：状态离散化精度依赖bins数，高维时效率低；易陷入局部最优。
- **改进建议**：引入epsilon decay、增加bins精度，或迁移到DQN处理连续状态。

## 结论
实验成功实现CartPole-v1的Q-learning，参数调整关键。未来可扩展到更复杂环境。

## 参考
- Gymnasium文档
- Q-learning论文