import os
import time
try:
    import gymnasium as gym
except Exception:
    import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


def q_learning(env, num_episodes=500, alpha=0.2, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.9985, min_epsilon=0.00001,
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


def sarsa(env, num_episodes=500, alpha=0.18, gamma=0.99,
          epsilon=1.0, epsilon_decay=0.9985, min_epsilon=0.0001,
          max_steps_per_episode=1000):
    """On-policy SARSA implementation. Returns Q and rewards list."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards_all_episodes = []

    for ep in range(num_episodes):
        # reset (handle tuple return)
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            state = reset_ret[0]
        else:
            state = reset_ret

        # choose initial action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
            step_ret = env.step(action)
            if len(step_ret) == 4:
                next_state, reward, done, info = step_ret
                next_action = None
            else:
                next_state, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)

            # choose next action for SARSA
            if not done:
                if np.random.rand() < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = int(np.argmax(Q[next_state]))
            else:
                next_action = 0  # value won't matter when done, but ensure int

            # SARSA update
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        # decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_all_episodes.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_all_episodes[-100:])
            print(f"SARSA Episode {ep+1}/{num_episodes} - avg(last100) reward: {avg:.2f} - eps: {epsilon:.3f}")

    return Q, rewards_all_episodes


def visualize_episode(env, Q, n_rows=4, n_cols=12, save_path=None, title="Agent Episode"):
    """
    Visualize an episode with the learned policy using matplotlib.
    Shows agent movement in real-time on the CliffWalking grid.
    CliffWalking coordinates: state 36 = bottom-left start, state 47 = bottom-right goal
    """
    # Reset environment
    reset_ret = env.reset()
    if isinstance(reset_ret, tuple):
        state = reset_ret[0]
    else:
        state = reset_ret
    
    # Setup the plot - use standard matplotlib coordinates (bottom-left origin)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_aspect('equal')
    # Don't invert y-axis, keep bottom-left as (0,0) to match CliffWalking convention
    
    # Draw grid
    for i in range(n_rows + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=1)
    for j in range(n_cols + 1):
        ax.axvline(x=j - 0.5, color='black', linewidth=1)
    
    # Color the cliff (states 37-46, bottom row except start and goal)
    cliff_states = list(range(n_cols * (n_rows - 1) + 1, n_cols * n_rows - 1))
    for cliff_state in cliff_states:
        # Convert state to grid coordinates: bottom-left is (0,0)
        state_row, state_col = cliff_state // n_cols, cliff_state % n_cols
        # For matplotlib: flip row coordinate (row 3 -> y=0, row 0 -> y=3)
        plot_row = (n_rows - 1) - state_row
        rect = patches.Rectangle((state_col - 0.5, plot_row - 0.5), 1, 1, 
                                linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.7)
        ax.add_patch(rect)
    
    # Mark start (state 36 = row 3, col 0 = bottom-left) and goal (state 47 = row 3, col 11 = bottom-right)
    start_state = 36  # Actual start state from environment
    goal_state = 47   # Actual goal state
    start_row, start_col = start_state // n_cols, start_state % n_cols
    goal_row, goal_col = goal_state // n_cols, goal_state % n_cols
    
    # Convert to matplotlib coordinates (flip row)
    start_plot_row = (n_rows - 1) - start_row
    goal_plot_row = (n_rows - 1) - goal_row
    
    start_rect = patches.Rectangle((start_col - 0.5, start_plot_row - 0.5), 1, 1,
                                  linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7)
    goal_rect = patches.Rectangle((goal_col - 0.5, goal_plot_row - 0.5), 1, 1,
                                 linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7)
    ax.add_patch(start_rect)
    ax.add_patch(goal_rect)
    
    # Agent representation - convert state to matplotlib coordinates
    agent_state_row, agent_col = state // n_cols, state % n_cols
    agent_plot_row = (n_rows - 1) - agent_state_row  # Flip for matplotlib
    agent_circle = plt.Circle((agent_col, agent_plot_row), 0.3, color='orange', zorder=10)
    ax.add_patch(agent_circle)
    
    # Path tracking
    path_x, path_y = [agent_col], [agent_plot_row]
    path_line, = ax.plot(path_x, path_y, 'o-', color='darkorange', linewidth=2, markersize=6, alpha=0.6)
    
    # Labels and title
    ax.set_title(f'{title} - Step 0', fontsize=14)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightgreen', label='Start'),
        patches.Patch(color='lightblue', label='Goal'),
        patches.Patch(color='lightcoral', label='Cliff'),
        patches.Patch(color='orange', label='Agent')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    # Collect episode data
    trajectory = [(agent_state_row, agent_col)]  # Store actual state coordinates
    actions_taken = []
    rewards_received = []
    
    done = False
    step_count = 0
    total_reward = 0
    
    while not done and step_count < 100:  # Safety limit
        # Choose action using learned policy
        action = int(np.argmax(Q[state]))
        actions_taken.append(action)
        
        # Step
        step_ret = env.step(action)
        if len(step_ret) == 4:
            next_state, reward, done, info = step_ret
        else:
            next_state, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        
        rewards_received.append(reward)
        total_reward += reward
        
        # Update position - convert state to coordinates
        agent_state_row, agent_col = next_state // n_cols, next_state % n_cols
        agent_plot_row = (n_rows - 1) - agent_state_row  # Flip for matplotlib
        trajectory.append((agent_state_row, agent_col))  # Store actual coordinates
        
        # Update visualization
        agent_circle.center = (agent_col, agent_plot_row)
        path_x.append(agent_col)
        path_y.append(agent_plot_row)
        path_line.set_data(path_x, path_y)
        
        action_names = ['^', '>', 'v', '<']
        ax.set_title(f'{title} - Step {step_count + 1} | Action: {action_names[action]} | Reward: {reward} | Total: {total_reward}', fontsize=14)
        
        plt.pause(0.5)  # Pause for animation effect
        
        state = next_state
        step_count += 1
    
    # Final update
    ax.set_title(f'{title} - FINISHED | Total Steps: {step_count} | Total Reward: {total_reward}', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved episode visualization to {save_path}")
    
    plt.show()
    
    return trajectory, actions_taken, rewards_received, total_reward


def test_policies_with_gui(env, Q_q, Q_s, n_rows=4, n_cols=12, out_dir='out'):
    """
    Test both Q-learning and SARSA policies with GUI visualization.
    """
    print("Testing Q-learning policy with GUI...")
    traj_q, actions_q, rewards_q, total_q = visualize_episode(
        env, Q_q, n_rows, n_cols, 
        save_path=os.path.join(out_dir, 'q_learning_episode.png'),
        title="Q-learning Policy"
    )
    
    print("Testing SARSA policy with GUI...")
    # Reset environment for SARSA test
    env.reset()
    traj_s, actions_s, rewards_s, total_s = visualize_episode(
        env, Q_s, n_rows, n_cols,
        save_path=os.path.join(out_dir, 'sarsa_episode.png'), 
        title="SARSA Policy"
    )
    
    # Print summary
    print("\n=== Episode Summary ===")
    print(f"Q-learning: {len(traj_q)} steps, total reward: {total_q}")
    print(f"SARSA: {len(traj_s)} steps, total reward: {total_s}")
    
    return {
        'q_learning': {'trajectory': traj_q, 'actions': actions_q, 'rewards': rewards_q, 'total_reward': total_q},
        'sarsa': {'trajectory': traj_s, 'actions': actions_s, 'rewards': rewards_s, 'total_reward': total_s}
    }


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


def plot_rewards(rewards, save_path=None, title=None):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label='episode reward')
    # moving average
    window = 50
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, window - 1 + len(ma)), ma, label=f'{window}-ep MA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    if title:
        plt.title(title)
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
        print("Make sure the correct version of gym/gymnasium is installed.\n")
        raise

    np.random.seed(0)

    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)

    # training hyperparameters
    num_episodes = 2000
    alpha = 0.23
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.99
    min_epsilon = 0.0001

    # Run Q-learning
    print("Starting Q-learning...")
    start = time.time()
    Q_q, rewards_q = q_learning(env, num_episodes=num_episodes, alpha=alpha, gamma=gamma,
                                epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
    dur_q = time.time() - start
    print(f"Q-learning finished in {dur_q:.1f}s - avg(last100): {np.mean(rewards_q[-100:]):.2f}")

    # Run SARSA
    print("Starting SARSA...")
    # create a fresh environment instance for SARSA to avoid state carry-over
    env_sarsa = gym.make(env_name)
    start = time.time()
    Q_s, rewards_s = sarsa(env_sarsa, num_episodes=num_episodes, alpha=alpha, gamma=gamma,
                            epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
    dur_s = time.time() - start
    print(f"SARSA finished in {dur_s:.1f}s - avg(last100): {np.mean(rewards_s[-100:]):.2f}")

    # policies
    policy_q = extract_policy(Q_q)
    policy_s = extract_policy(Q_s)

    # print policies
    try:
        if hasattr(env.unwrapped, 'shape'):
            n_rows, n_cols = env.unwrapped.shape
        else:
            n_rows, n_cols = 4, 12
    except Exception:
        n_rows, n_cols = 4, 12

    print("Q-learning policy:")
    print_policy(policy_q, n_rows=n_rows, n_cols=n_cols)
    print("SARSA policy:")
    print_policy(policy_s, n_rows=n_rows, n_cols=n_cols)

    # Test policies with GUI visualization
    print("\n" + "="*50)
    print("TESTING POLICIES WITH GUI VISUALIZATION")
    print("="*50)
    
    test_results = test_policies_with_gui(env, Q_q, Q_s, n_rows, n_cols, out_dir)

    # plots: separate and combined
    plot_rewards(rewards_q, save_path=os.path.join(out_dir, 'rewards_q_learning.png'), title='Q-learning Training Rewards')
    plt.show()
    plot_rewards(rewards_s, save_path=os.path.join(out_dir, 'rewards_sarsa.png'), title='SARSA Training Rewards')
    plt.show()

    # combined plot
    plt.figure(figsize=(10, 5))
    episodes = np.arange(1, len(rewards_q) + 1)
    plt.plot(episodes, rewards_q, alpha=0.3, label='Q-learning reward')
    plt.plot(episodes, rewards_s, alpha=0.3, label='SARSA reward')
    # moving averages
    window = 50
    if len(rewards_q) >= window:
        ma_q = np.convolve(rewards_q, np.ones(window) / window, mode='valid')
        ma_s = np.convolve(rewards_s, np.ones(window) / window, mode='valid')
        plt.plot(range(window, window + len(ma_q)), ma_q, label=f'Q-learning {window}-ep MA')
        plt.plot(range(window, window + len(ma_s)), ma_s, label=f'SARSA {window}-ep MA')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-learning vs SARSA on CliffWalking-v0')
    plt.legend()
    plt.grid(True)
    combined_file = os.path.join(out_dir, 'rewards_compare.png')
    plt.tight_layout()
    plt.savefig(combined_file)
    print(f"Saved combined reward plot to {combined_file}")
    plt.show()

    # save Q-tables
    np.save(os.path.join(out_dir, 'q_table_q_learning.npy'), Q_q)
    np.save(os.path.join(out_dir, 'q_table_sarsa.npy'), Q_s)
    print(f"Saved Q-tables to {out_dir}")
    
    print("\nAll training and testing completed!")
    print(f"Check {out_dir} for all generated files:")
    print("- Reward plots (individual and comparison)")
    print("- Q-tables (.npy files)")
    print("- Episode visualizations (.png files)")


if __name__ == '__main__':
    main()
