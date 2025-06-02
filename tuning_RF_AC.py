import os
import csv
import itertools
import time

import subprocess
import ast
# Use sys module to access the current Python interpreter path across different operating systems
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", choices=["REINFORCE","ActorCritic"], default="REINFORCE")
args = parser.parse_args()

# --- HYPERPARAMETER GRID ---
gammas = [0.95, 0.99, 0.995]
hiddens = [32, 64, 128]
lrs_policy = [1e-4, 5e-4, 1e-3]
lrs_critic = [1e-4, 5e-4, 1e-3]
baselines = [0.0] # insert only a single value if you're using ActorCritic
AC_critic = ['V'] # insert only a single value if you're using REINFORCE

# Numero di episodi per run (puoi abbassarlo in fase di test)
N_EPISODES = 10000
PRINT_EVERY = 2000
ALG = args.algorithm
DEVICE = "cpu"
# BASELINE = "dynamic"

 # Build the command-line arguments list to run train.py:
 # - Starts with the current Python interpreter in unbuffered mode
 # - Appends each flag and its value from args_dict
def get_args(args_dict):
    # Use sys.executable to invoke the same interpreter running this script, ensuring cross-platform compatibility
    # Initialize args list with interpreter and script
    args = [sys.executable, "-u", "train.py"]
    # Append each argument flag and its corresponding value
    for key, value in args_dict.items():
        args.append(key)
        args.append(str(value))
    # Return the complete list of command-line arguments
    return args

# Parse the subprocess output to extract the final return value printed by train.py
def get_final_return(result):
    # Split the output text into lines and search for the line starting with "FINAL_RESULT:"
    for line in result.strip().split('\n'):
        # Identify the line containing the final result
        if line.startswith("FINAL_RESULT:"):
            
            final_line = line.split("FINAL_RESULT:")[1].strip()
            # Convert the extracted text to a Python literal (e.g., int or float)
            return ast.literal_eval(final_line)
    
    # If no final result line is found, output the full stdout and stderr for debugging
    print("FINAL_RESULT not found. Full output:")
    print(result.stdout)
    print("Error output:")
    print(result.stderr)
    # Return None to indicate failure to parse a result
    return None

# Execute the training process, streaming its output live to console and capturing it
def run_with_live_output(args):
    # Start subprocess with unified stdout and stderr in text mode
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    output_lines = []
    # Read each output line as it is produced, print it, and store it
    for line in process.stdout:
        print(line, end="")  # Print to console in real time
        output_lines.append(line)

    # Wait for the subprocess to complete
    process.wait()
    # Combine all captured lines into a single string for parsing
    return ''.join(output_lines)

# --- Save tuning results ---
# Create the output directory if it doesn't exist
os.makedirs("tuning_results", exist_ok=True)

csv_path = f"tuning_results/summary_{ALG}.csv"
file_exists = os.path.isfile(csv_path)

# Open a new CSV file for this algorithm's summary (overwrite if exists)
with open(csv_path, "a", newline="") as fout:
    # Determine CSV columns: include baseline column only when using a dynamic baseline
    if ALG == "REINFORCE":
        fieldnames = [
            "gamma", "hidden", "lr_policy", "lr_critic", "baseline", "final_return", "elapsed_time"
            ]
    else:
        fieldnames = [
            "critic", "gamma", "hidden", "lr_policy", "lr_critic", "final_return", "elapsed_time"
            ]
    # Initialize CSV writer with the selected columns
    writer = csv.DictWriter(fout, fieldnames=fieldnames)

    # Write the header only if the file is new or empty
    if not file_exists or os.path.getsize(csv_path) == 0:
        writer.writeheader()

    # Iterate over every combination of discount factor, hidden layer size, and learning rates to evaluate agent performance
    for AC_critic, gamma, hidden, lr_policy, lr_critic, baseline in itertools.product(AC_critic, gammas, hiddens, lrs_policy, lrs_critic, baselines):

        # Log the current hyperparameter configuration to console
        print(f"[γ={gamma:.3f}, h={hidden}, lr_policy={lr_policy:.1e}, lr_critic={lr_critic:.1e}, baseline={baseline}, AC_critic={AC_critic}]")

        # Construct a dictionary of command-line flags and values for this trial
        args_dict = {
            '--algorithm' : ALG,
            '--n-episodes': N_EPISODES,
            '--print-every' : PRINT_EVERY,
            '--device' : DEVICE,
            '--baseline': baseline,
            '--gamma': gamma,
            '--lr-policy': lr_policy,
            '--lr-critic': lr_critic,
            '--hidden': hidden,
            '--AC-critic': AC_critic
        }

        # Generate the complete argument list to launch train.py with the specified parameters
        args = get_args(args_dict)

        start_time = time.time()

        # Execute the subprocess, streaming output live and capturing it for parsing
        result = run_with_live_output(args)

        # Parse the captured output to extract the numeric final return value
        final_return = get_final_return(result)
        elapsed_time  = time.time() - start_time
        
        # Scrivi nel CSV
        if ALG == "REINFORCE":
            row = {
            "gamma": gamma,
            "hidden": hidden,
            "lr_policy": lr_policy,
            "lr_critic": lr_critic,
            "baseline" : baseline,
            "final_return": final_return,
            "elapsed_time": elapsed_time
        }
        else:
            row = {
            "critic": AC_critic,
            "gamma": gamma,
            "hidden": hidden,
            "lr_policy": lr_policy,
            "lr_critic": lr_critic,
            "final_return": final_return,
            "elapsed_time": elapsed_time
        }
            
        writer.writerow(row)

print("Tuning completato. Risultati in tuning_results/summary.csv")