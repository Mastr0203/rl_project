import os
import csv
import itertools
import subprocess

# --- HYPERPARAMETER GRID ---
algos = ["PPO", "SAC"]
lrs = [1e-4, 3e-4, 5e-4]
batch_sizes = [64, 128, 256]
n_steps_list = [1024, 2048, 4096]  # solo per PPO

TOTAL_TIMESTEPS = 500_000
TRAIN_DOMAIN = "source"
TEST_DOMAIN = "target"

def get_args_dict(algo, lr, batch_size, n_steps=None):
    args = {
        '--algo': algo,
        '--train_domain': TRAIN_DOMAIN,
        '--test_domain': TEST_DOMAIN,
    }
    return args, {
        "TOTAL_TIMESTEPS": str(TOTAL_TIMESTEPS),
        "LEARNING_RATE": str(lr),
        "BATCH_SIZE": str(batch_size),
        "N_STEPS": str(n_steps) if n_steps is not None else ""
    }

def get_args_list(args_dict):
    args = ["python", "train_sb3.py"]
    for key, value in args_dict.items():
        args.append(key)
        args.append(str(value))
    return args

def run_process(args, env_vars):
    env = os.environ.copy()
    env.update({k: v for k, v in env_vars.items() if v})  # rimuove env vuote

    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines = []
    for line in process.stdout:
        print(line, end="")  # stampa live
        output_lines.append(line)
    process.wait()
    return ''.join(output_lines)

def extract_reward(output):
    for line in output.splitlines():
        if "Final evaluation" in line and "mean_reward=" in line:
            try:
                return float(line.split("mean_reward=")[1].split(" +/- ")[0])
            except Exception:
                return None
    return None

# --- CSV Output ---
os.makedirs("tuning_results", exist_ok=True)
with open("tuning_results/summary_SB3.csv", "w", newline="") as fout:
    fieldnames = ["algo", "learning_rate", "batch_size", "n_steps", "final_reward"]
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for algo in algos:
        for lr, bs in itertools.product(lrs, batch_sizes):
            if algo == "PPO":
                for n_steps in n_steps_list:
                    print(f"[{algo}] lr={lr} | bs={bs} | n_steps={n_steps}")
                    args_dict, env_vars = get_args_dict(algo, lr, bs, n_steps)
                    args_list = get_args_list(args_dict)
                    output = run_process(args_list, env_vars)
                    final_reward = extract_reward(output)
                    writer.writerow({
                        "algo": algo,
                        "learning_rate": lr,
                        "batch_size": bs,
                        "n_steps": n_steps,
                        "final_reward": final_reward
                    })
            else:  # SAC
                print(f"[{algo}] lr={lr} | bs={bs}")
                args_dict, env_vars = get_args_dict(algo, lr, bs)
                args_list = get_args_list(args_dict)
                output = run_process(args_list, env_vars)
                final_reward = extract_reward(output)
                writer.writerow({
                    "algo": algo,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "n_steps": None,
                    "final_reward": final_reward
                })

print("\nTuning completato. Risultati in tuning_results/summary_SB3.csv")