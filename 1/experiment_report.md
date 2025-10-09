# CartPole-v1 强化学习实验报告

## 摘要
本实验使用Q-learning算法实现CartPole-v1环境的平衡任务，通过离散化状态空间、epsilon-greedy策略和epsilon衰减机制训练智能体。实验结果显示，在优化参数下，智能体能在10000次训练episode内达到平均奖励约102，并通过数据表格、图线可视化和图形化演示展示训练过程，证明算法的有效性和稳定性。

## 引言
CartPole-v1是经典的强化学习基准任务，要求智能体通过左右推力保持杆子平衡。状态包括小车位置、速度、杆子角度和角速度。奖励为每步1分，episode在杆倒或超出界限时终止。

本实验采用Q-learning，结合状态离散化和epsilon衰减，探索参数对收敛和稳定性的影响。相比传统方法，引入epsilon decay以平衡探索与利用，并启用图形化演示以直观展示智能体性能。

## 方法
### 算法概述
- **Q-learning**：时序差分学习，更新公式：  
  `Q(s, a) += α * (r + γ * max(Q(s')) - Q(s, a))`
- **状态离散化**：将连续状态映射到离散bins，使用`np.linspace`定义边界，bins数为(18, 18, 18, 18)，总状态数约1.05e6。
- **epsilon-greedy + decay**：初始epsilon=1.0，以epsilon_decay=0.9991逐步衰减至epsilon_min=0.00001，实现从探索到利用的平滑过渡。

### 代码实现
- **类QLearning**：封装初始化、训练、测试方法。
- **训练流程**：每1000 episode测试15次，记录平均奖励；epsilon每episode衰减。
- **终止条件**：步数>200、位置>2.4、角度>41.8°。
- **数据处理**：使用pandas汇总训练数据到表格，并用matplotlib绘制奖励曲线；测试时启用渲染以图形化演示智能体行为。
- **可视化**：生成training_plot.png显示奖励曲线；终端打印pandas DataFrame表格；渲染窗口展示实时平衡过程。

#### 关键函数讲解
- **`__init__(self, bins, gamma, alpha, epsilon, epsilon_decay, epsilon_min, num_episodes)`**：初始化类参数，包括状态离散化bins、折扣因子gamma、学习率alpha、初始epsilon、衰减率epsilon_decay、最小epsilon和训练episode数。创建Q表（形状为bins + (2,)，初始化为0），并初始化Gymnasium环境。
- **`train(self)`**：执行训练循环。对于每个episode，重置环境，离散化初始状态，选择动作（epsilon-greedy），执行动作获取奖励和下一状态，更新Q值（Q-learning公式），衰减epsilon。每1000 episode调用test()记录平均奖励，并收集数据用于后续汇总。
- **`test(self, num_episodes=15)`**：执行测试循环，禁用epsilon（纯贪婪策略），运行指定episode数，计算平均奖励。启用渲染以可视化智能体行为，并返回平均奖励用于训练监控。

## 实验设置
- **环境**：Gymnasium CartPole-v1。
- **参数**：
  - bins: (18, 18, 18, 18)
  - gamma: 0.99
  - alpha: 0.18
  - epsilon: 1.0 (初始)
  - epsilon_decay: 0.9991
  - epsilon_min: 0.00001
  - num_episodes: 10000
  - 测试episode数: 15
- **硬件**：标准PC，Python 3.12，依赖：gymnasium, numpy, matplotlib, pandas。

## 结果
- **训练曲线**：奖励随episode增加波动上升，早期探索导致低奖励，中期收敛至90-110区间，后期稳定但有轻微波动（见training_plot.png）。
- **性能指标**：10000 episode时平均奖励约101.8，最高达105左右（后期episode），表明策略基本收敛但未达到200（CartPole最优）。
- **数据汇总**：表格显示每1000 episode的平均奖励，从14.2（初始）逐步提升至101.8，反映epsilon decay的效果。
- **可视化**：生成training_plot.png显示奖励vs episode曲线；终端打印pandas DataFrame表格，便于分析；图形化演示窗口展示智能体在测试episode中的平衡过程。

## 讨论
- **优势**：epsilon decay显著提升稳定性，避免后期探索不足；离散化简单有效，无需深度网络；表格+图线+渲染展示便于调试、报告和演示。
- **局限**：离散化精度有限（18 bins仍粗糙），导致未达到最优奖励；Q-table内存占用大（1M+状态）；epsilon_min过低可能限制最终微调；渲染可能影响性能但增强可视化。
- **参数影响**：alpha=0.18较高，促进快速学习但易震荡；epsilon_decay=0.9991平衡探索/利用；bins增加提升精度但计算成本高；测试episode数15提供更精确平均但增加计算时间。
- **改进建议**：进一步增加bins或episode数；对比DQN（连续状态处理）；添加经验回放或目标网络；优化渲染以减少延迟。

## 结论
实验成功实现CartPole-v1的Q-learning with epsilon decay，参数调整关键于收敛速度和稳定性。数据表格、图线和图形化演示增强了实验可复现性和教育价值。未来可扩展到策略迭代或DQN，以处理更复杂环境。

## 参考
- Gymnasium文档 (https://gymnasium.farama.org/)
- Q-learning原始论文 (Watkins, 1989)
- Sutton & Barto《Reinforcement Learning: An Introduction》