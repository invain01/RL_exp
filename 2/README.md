# Q-learning for CliffWalking-v0

This folder contains a simple Q-learning implementation for the CliffWalking environment.

Files:
- `q_learning_cliffwalking.py`: training script (trains and saves Q-table and reward plot to `out/`).
- `requirements.txt`: minimal dependencies (uses `gymnasium`).

Run (Windows PowerShell):

```powershell
# from repository root
C:/Path/To/Python/python.exe "2\q_learning_cliffwalking.py"
```

Notes:
- The script prefers `gymnasium` (maintained fork of `gym`) to avoid compatibility issues with newer NumPy.
- Trained artifacts are saved in `2/out/`.
