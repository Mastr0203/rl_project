import os
import csv
import itertools

import subprocess
import ast

# --- HYPERPARAMETER GRID ---
gammas = [0.95, 0.99, 0.995]
hiddens = [32, 64, 128]
lrs_policy = [1e-4, 5e-4, 1e-3]
lrs_critic = [1e-4, 5e-4, 1e-3]

# Numero di episodi per run (puoi abbassarlo in fase di test)
N_EPISODES = 1000
PRINT_EVERY = 200
ALG = "REINFORCE"  # o "REINFORCE"
DEVICE = "cpu"
BASELINE = "dynamic"

def get_args(args_dict):
    args = args = args = [os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe"), "-u", "train.py"]
    for key, value in args_dict.items():
        args.append(key)
        args.append(str(value))
    return args

def get_final_return(result):
    for line in result.strip().split('\n'):
        if line.startswith("FINAL_RESULT:"):
            
            final_line = line.split("FINAL_RESULT:")[1].strip()
            return ast.literal_eval(final_line)
    
    print("FINAL_RESULT not found. Full output:")
    print(result.stdout)
    print("Error output:")
    print(result.stderr)
    return None

def run_with_live_output(args):
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    output_lines = []
    for line in process.stdout:
        print(line, end="")  # Print to console in real time
        output_lines.append(line)

    process.wait()
    return ''.join(output_lines)

# Salva risultati
os.makedirs("tuning_results", exist_ok=True)
with open(f"tuning_results/summary_{ALG}.csv", "w", newline="") as fout:

    if ALG == "REINFORCE" and BASELINE != 0:
        fieldnames = [
        "gamma", "hidden", "lr_policy", "lr_critic", "baseline", "final_return"
    ]
    else:
        fieldnames = [
        "gamma", "hidden", "lr_policy", "lr_critic", "final_return"
    ]
        
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    # Loop sulle combinazioni
    for gamma, hidden, lr_policy, lr_critic in itertools.product(gammas, hiddens, lrs_policy, lrs_critic):

        print(f"[γ={gamma:.3f} h={hidden} lr_policy={lr_policy:.1e} lr_critic={lr_critic:.1e}]")

        args_dict = {
            '--algorithm' : ALG,
            '--n-episodes': N_EPISODES,
            '--print-every' : PRINT_EVERY,
            '--device' : DEVICE,
            '--baseline': BASELINE,
            '--gamma': gamma,
            '--lr-policy': lr_policy,
            '--lr-critic': lr_critic,
            '--hidden': hidden
        }

        args = get_args(args_dict)

        result = run_with_live_output(args)

        final_return = get_final_return(result)
        
        # Scrivi nel CSV
        if ALG == "REINFORCE" and BASELINE != 0:
            row = {
            "gamma": gamma,
            "hidden": hidden,
            "lr_policy": lr_policy,
            "lr_critic": lr_critic,
            "baseline" : BASELINE,
            "final_return": final_return
        }
        else:
            row = {"gamma": gamma,
            "hidden": hidden,
            "lr_policy": lr_policy,
            "lr_critic": lr_critic,
            "final_return": final_return
        }
            
        writer.writerow(row)

print("Tuning completato. Risultati in tuning_results/summary.csv")