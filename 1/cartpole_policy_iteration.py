import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, env, bins=(10, 10, 10, 10), gamma=0.99, alpha=0.1, epsilon=0.1, num_episodes=50000):
        self.env = env
        self.bins = bins
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_states = np.prod(bins)
        self.num_actions = env.action_space.n
        self.Q = np.zeros((self.num_states, self.num_actions))

    def discretize_state(self, state):
        position_bins, velocity_bins, angle_bins, angular_velocity_bins = self.bins
        discretized = [
            np.clip(np.digitize(state[0], np.linspace(-4.8, 4.8, position_bins)) - 1, 0, position_bins - 1),
            np.clip(np.digitize(state[1], np.linspace(-3.0, 3.0, velocity_bins)) - 1, 0, velocity_bins - 1),
            np.clip(np.digitize(state[2], np.linspace(-0.418, 0.418, angle_bins)) - 1, 0, angle_bins - 1),
            np.clip(np.digitize(state[3], np.linspace(-2.0, 2.0, angular_velocity_bins)) - 1, 0, angular_velocity_bins - 1)
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

    def train(self):
        rewards = []
        episodes = []
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            state_idx = self.state_to_index(state)
            done = False
            step_count = 0
            while not done:
                action = self.choose_action(state_idx)
                next_state, reward, done, _, _ = self.env.step(action)
                step_count += 1
                # 检查自定义终止条件
                if step_count > 200 or abs(next_state[0]) > 2.4 or abs(next_state[2]) > np.radians(12):
                    done = True
                next_state = self.discretize_state(next_state)
                next_state_idx = self.state_to_index(next_state)
                self.Q[state_idx, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state_idx]) - self.Q[state_idx, action])
                state_idx = next_state_idx
            if episode % 1000 == 0:
                print(f"Episode {episode}")
                # 测试
                test_env = gym.make('CartPole-v1', render_mode='human')
                original_env = self.env
                self.env = test_env
                avg_reward = self.test(num_episodes=10, no_render=False, verbose=True)
                rewards.append(avg_reward)
                episodes.append(episode)
                print(f"After {episode} episodes, average reward: {avg_reward}")
                self.env = original_env
                test_env.close()
        return episodes, rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def test(self, num_episodes=10, no_render=False, verbose=True):
        policy = self.get_policy()
        total_rewards = []
        for i in range(num_episodes):
            if verbose:
                print(f"Test episode: {i+1}")
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            done = False
            episode_reward = 0
            step_count = 0
            while not done:
                action = policy[self.state_to_index(state)]
                next_state, reward, done, _, _ = self.env.step(action)
                step_count += 1
                if verbose:
                    print(f"Position: {next_state[0]:.2f}, Velocity: {next_state[1]:.2f}, Angle: {next_state[2]:.2f}, Angular Velocity: {next_state[3]:.2f}", flush=True)
                if not no_render:
                    self.env.render()
                    time.sleep(0.05)
                episode_reward += reward
                # 检查自定义终止条件
                if step_count > 200 or abs(next_state[0]) > 2.4 or abs(next_state[2]) > np.radians(12):
                    done = True
                state = self.discretize_state(next_state)
            if verbose:
                print("Episode ended")
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

if __name__ == "__main__":
    env_train = gym.make('CartPole-v1')
    ql = QLearning(env_train, num_episodes=5000)
    episodes, rewards = ql.train()
    ql.env = gym.make('CartPole-v1', render_mode='human')  # 切换到render env for final test
    avg_reward = ql.test(num_episodes=10)
    print(f"Final average reward: {avg_reward}")
    ql.env.close()
    
    # 绘制图像
    plt.plot(episodes, rewards)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Training Episodes')
    #plt.savefig('training_plot.png')
    plt.show()