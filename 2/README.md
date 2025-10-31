# Q-learning for CliffWalking-v0

This folder contains implementations for both Q-learning and SARSA on the CliffWalking environment.

Files:
- `q_learning_cliffwalking.py`: training script that runs both Q-learning and SARSA, prints policies, saves Q-tables and reward plots to `out/`, and includes GUI visualization of policy testing.
- `requirements.txt`: minimal dependencies (uses `gymnasium`).

Run (Windows PowerShell):

```powershell
# from repository root (adjust path to your Python if needed)
C:/Path/To/Python/python.exe "2\q_learning_cliffwalking.py"
```

Outputs (in `2/out/`):
- `rewards_q_learning.png` — reward curve for Q-learning
- `rewards_sarsa.png` — reward curve for SARSA
- `rewards_compare.png` — combined comparison plot (episode rewards + moving averages)
- `q_table_q_learning.npy`, `q_table_sarsa.npy` — saved Q-tables
- `q_learning_episode.png` — GUI visualization of Q-learning policy execution
- `sarsa_episode.png` — GUI visualization of SARSA policy execution

Notes:
- The script prefers `gymnasium` (maintained fork of `gym`) to avoid compatibility issues with newer NumPy.
- If you want to adjust training hyperparameters (episodes, alpha, etc.), edit the variables near the top of `main()` in the script.
- The GUI visualization shows the agent (orange circle) moving through the grid, with:
  - Green: Start position
  - Blue: Goal position  
  - Red: Cliff areas (dangerous)
  - Orange path: Agent's trajectory
- Each step displays the action taken, immediate reward, and cumulative reward.
