import os
import time
try:
    import gymnasium as gym
except Exception:
    import gym
import numpy as np
import matplotlib.pyplot as plt


def q_learning(env, num_episodes=500, alpha=0.5, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01,
               max_steps_per_episode=1000):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards_all_episodes = []

    for ep in range(num_episodes):
        # support both old gym.reset() -> obs and gymnasium.reset() -> (obs, info)
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            state = reset_ret[0]
        else:
            state = reset_ret
        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            # support old gym.step() -> (obs, reward, done, info)
            # and gymnasium.step() -> (obs, reward, terminated, truncated, info)
            step_ret = env.step(action)
            if len(step_ret) == 4:
                next_state, reward, done, info = step_ret
            else:
                next_state, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)

            # Q-learning update
            best_next = np.max(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * best_next - Q[state, action]
            )

            state = next_state
            total_reward += reward

            if done:
                break

        # decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_all_episodes.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_all_episodes[-100:])
            print(f"Episode {ep+1}/{num_episodes} - avg(last100) reward: {avg:.2f} - eps: {epsilon:.3f}")

    return Q, rewards_all_episodes


def extract_policy(Q):
    return np.argmax(Q, axis=1)


def print_policy(policy, n_rows=4, n_cols=12):
    # CliffWalking default: 4 rows x 12 cols (state numbering row-major)
    action_map = {0: '^', 1: '>', 2: 'v', 3: '<'}
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            s = r * n_cols + c
            row.append(action_map.get(int(policy[s]), '?'))
        grid.append(row)

    print("Policy (rows top->bottom):")
    for row in grid:
        print(' '.join(row))


def plot_rewards(rewards, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label='episode reward')
    # moving average
    window = 50
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, window - 1 + len(ma)), ma, label=f'{window}-ep MA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved reward plot to {save_path}")
    else:
        plt.show()


def main():
    env_name = 'CliffWalking-v0'
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Failed to create env '{env_name}': {e}")
        print("Make sure the correct version of gym is installed (classic 'gym' package with toy_text).\n")
        raise

    np.random.seed(0)
    try:
        env.seed(0)
    except Exception:
        pass

    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()
    Q, rewards = q_learning(env, num_episodes=1000, alpha=0.7, gamma=0.99,
                            epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)
    duration = time.time() - start

    print(f"Training finished in {duration:.1f}s")
    print(f"Average reward (last 100): {np.mean(rewards[-100:]):.2f}")

    policy = extract_policy(Q)

    # print policy with arrows
    try:
        # Infer grid size if possible from env (CliffWalking specifics)
        if hasattr(env, 'shape'):
            n_rows, n_cols = env.shape
        else:
            n_rows, n_cols = 4, 12
    except Exception:
        n_rows, n_cols = 4, 12

    print_policy(policy, n_rows=n_rows, n_cols=n_cols)

    plot_file = os.path.join(out_dir, 'rewards.png')
    plot_rewards(rewards, save_path=plot_file)

    # also save Q-table
    np.save(os.path.join(out_dir, 'q_table.npy'), Q)
    print(f"Saved Q-table to {os.path.join(out_dir, 'q_table.npy')}")


if __name__ == '__main__':
    main()
