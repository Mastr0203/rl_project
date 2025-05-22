import itertools
#from train import train_model as train_custom
from train_sb3 import train_model as train_sb3
from train_sb3 import make_model_name
import argparse
import csv
from datetime import datetime


parser = argparse.ArgumentParser(description="Train PPO or SAC on CustomHopper")
parser.add_argument("--algo", choices=["PPO", "SAC"], default="PPO",
                        help="Algorithm to use: PPO or SAC")
parser.add_argument("--train_domain", choices=["source", "target"], default="source", help="Domain to train on [source, target]")
parser.add_argument("--test_domain", choices=["source", "target"], default="target", help="Domain to test on [source, target]")

args = parser.parse_args()
algo = args.algo

param_spaces = {
    "REINFORCE": {
        "lr": [1e-4, 5e-4, 1e-3],
        "gamma": [0.95, 0.99, 0.995]
        # aggiunger cose...
    },
    "ActorCritic": {
        "lr_actor": [1e-4, 5e-4, 1e-3],
        "lr_critic": [1e-4, 5e-4, 1e-3],
        "gamma": [0.95, 0.99, 0.995]
        # aggiunger cose...
    },
    "PPO": {
        "learning_rate": [1e-4, 3e-4, 5e-4],
        "batch_size": [64, 128, 256],
        "n_steps": [1024, 2048, 4096],
        "ent_coef": [0, 0.01, 0.05, 0.1]

    },
    "SAC": {
        "learning_rate": [1e-4, 3e-4, 5e-4],
        "batch_size": [128, 256],
        "gradient_steps": [1,4],
        "learning_starts": [5e3,1e4],
        "buffer_size": [1e5,1e6],
        "policy_kwargs": [dict(net_arch=[128, 128]), dict(net_arch=[256, 256])]
    }
}

def grid_search(algo, param_grid, train_domain, test_domain,total_timesteps):
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"tuning_results/summary_{algo}_{timestamp}.csv"
    model_filename = make_model_name(algo,args.train_domain,args.test_domain, udr=False)

    with open(csv_filename, "w", newline="") as f:
        fieldnames = list(param_grid.keys()) + ["reward", "std_reward"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(all_combinations):
            print(f"Trial {i+1}/{len(all_combinations)}: {combo}")
    #       if algo in ["REINFORCE", "ActorCritic"]:
    #           reward = train_custom(algo, combo, env)
            if algo in ["PPO", "SAC"]:
                if "buffer_size" in combo:
                    combo["buffer_size"] = int(combo["buffer_size"])
                if "learning_starts" in combo:
                    combo["learning_starts"] = int(combo["learning_starts"])
                reward, std_reward= train_sb3(algo, combo, train_domain, test_domain,total_timesteps, model_filename,UDR = False, WandDB= False,)
            else:
                raise ValueError("Unsupported algorithm")

            print(f"→ Mean reward: {reward:.2f} +- {std_reward:.2f}")
            combo_reward = combo.copy()
            combo_reward["reward"] = reward
            combo_reward["std_reward"] = std_reward
            writer.writerow(combo_reward)
            results.append((combo, reward))

    return sorted(results, key=lambda x: x[1], reverse=True)

def main():
    total_timesteps = 500_000
    results = grid_search(algo, param_spaces[algo], args.train_domain, args.test_domain,total_timesteps)

    print("Top configs:")
    for combo, reward, std_reward in results[:3]:
        print(f"{combo} → Reward: {reward:.2f} +- {std_reward:.2f}")

if __name__ == "__main__":
    main()