import itertools
#from train import train_model as train_custom
from train_sb3 import train_model as train_sb3
import argparse
import csv


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
        "batch_size": [64, 128, 256]
    }
}

def grid_search(algo, param_grid, train_domain, test_domain,total_timesteps):
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = []

    csv_filename = f"tuning_results/summary_{algo}.csv"
    with open(csv_filename, "w", newline="") as f:
        fieldnames = list(param_grid.keys()) + ["reward"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(all_combinations):
            print(f"Trial {i+1}/{len(all_combinations)}: {combo}")
    #       if algo in ["REINFORCE", "ActorCritic"]:
    #           reward = train_custom(algo, combo, env)
            if algo in ["PPO", "SAC"]:
                reward, std_reward, model = train_sb3(algo, combo, train_domain, test_domain,total_timesteps, callbacks=None)
            else:
                raise ValueError("Unsupported algorithm")

            print(f"→ Mean reward: {reward:.2f}")
            combo_reward = combo.copy()
            combo_reward["reward"] = reward
            writer.writerow(combo_reward)
            results.append((combo, reward))

    return sorted(results, key=lambda x: x[1], reverse=True)

def main():
    total_timesteps = 10000
    results = grid_search(algo, param_spaces[algo], args.train_domain, args.test_domain,total_timesteps)

    print("Top configs:")
    for combo, reward in results[:3]:
        print(f"{combo} → Reward: {reward:.2f}")

if __name__ == "__main__":
    main()