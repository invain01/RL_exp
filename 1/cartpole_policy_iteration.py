import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, env, bins=(15, 15, 15, 15), gamma=0.99, alpha=0.2, epsilon=1, num_episodes=10000):
        self.env = env
        self.bins = bins
        self.position_bins = np.linspace(-2.4, 2.4, self.bins[0])
        self.velocity_bins = np.linspace(-5.0, 5.0, self.bins[1])
        self.angle_bins = np.linspace(np.radians(-41.8), np.radians(41.8), self.bins[2])
        self.angular_velocity_bins = np.linspace(-5.0, 5.0, self.bins[3])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = 0.9991  # 添加epsilon衰减
        self.epsilon_min = 0.00001
        self.num_episodes = num_episodes
        self.num_states = np.prod(bins)
        self.num_actions = env.action_space.n
        self.Q = np.zeros((self.num_states, self.num_actions))

    def train(self):
        rewards = []
        episodes = []
        for episode in range(self.num_episodes + 1):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            state_idx = self.state_to_index(state)
            done = False
            step_count = 0
            while not done:
                action = self.choose_action(state_idx)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                step_count += 1
                # 检查自定义终止条件
                if step_count > 200 or abs(next_state[0]) > 2.4 or abs(next_state[2]) > np.radians(12):
                    done = True
                next_state = self.discretize_state(next_state)
                next_state_idx = self.state_to_index(next_state)
                self.Q[state_idx, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state_idx]) - self.Q[state_idx, action])
                state_idx = next_state_idx
            # epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if episode % 1000 == 0:
                print(f"Episode {episode}")
                # 测试
                # test_env = gym.make('CartPole-v1', render_mode='human')
                test_env = gym.make('CartPole-v1')
                original_env = self.env
                self.env = test_env
                avg_reward = self.test(num_episodes=20, no_render=True, verbose=True)
                rewards.append(avg_reward)
                episodes.append(episode)
                print(f"After {episode} episodes, average reward: {avg_reward}")
                self.env = original_env
                test_env.close()
        return episodes, rewards

    def discretize_state(self, state):
        discretized = [
            np.clip(np.digitize(state[0], self.position_bins) - 1, 0, self.bins[0] - 1),
            np.clip(np.digitize(state[1], self.velocity_bins) - 1, 0, self.bins[1] - 1),
            np.clip(np.digitize(state[2], self.angle_bins) - 1, 0, self.bins[2] - 1),
            np.clip(np.digitize(state[3], self.angular_velocity_bins) - 1, 0, self.bins[3] - 1)
        ]
        return tuple(discretized)

    def state_to_index(self, state):
        index = 0
        multiplier = 1
        for i in range(len(state)):
            index += state[i] * multiplier
            multiplier *= self.bins[i]
        return index

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state_idx])

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def test(self, num_episodes=10, no_render=False, verbose=True):
        policy = self.get_policy()
        total_rewards = []
        for i in range(num_episodes):
            # if verbose:
            #     print(f"Test episode: {i+1}")
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            done = False
            episode_reward = 0
            step_count = 0
            while not done:
                action = policy[self.state_to_index(state)]
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                step_count += 1
                # if verbose:
                #     print(f"Position: {next_state[0]:.2f}, Velocity: {next_state[1]:.2f}, Angle: {next_state[2]:.2f}, Angular Velocity: {next_state[3]:.2f}", flush=True)
                # if not no_render:
                #     self.env.render()
                #     time.sleep(0.001)
                episode_reward += reward
                # 检查自定义终止条件
                if step_count > 200 or abs(next_state[0]) > 2.4 or abs(next_state[2]) > np.radians(12):
                    done = True
                state = self.discretize_state(next_state)
            # if verbose:
            #     print("Episode ended")
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

if __name__ == "__main__":
    env_train = gym.make('CartPole-v1')
    ql = QLearning(env_train, num_episodes=10000)
    episodes, rewards = ql.train()
    
    # 绘制图像
    plt.plot(episodes, rewards)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Training Episodes')
    plt.savefig('training_plot.png')
    plt.show()