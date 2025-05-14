import argparse
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from env.custom_hopper import *  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to model.zip")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"], help="RL algorithm used")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--dimension", default="target", choices=["source", "target"], help="Domain to test on [source, target]")
    args = parser.parse_args()

    #  Carica il modello SB3
    if args.algo == "PPO":
        model = PPO.load(args.model)
    elif args.algo == "SAC":
        model = SAC.load(args.model)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    #  Crea l’ambiente target
    env = gym.make(f"CustomHopper-{args.dimension}-v0", render_mode="human" if args.render else None)

    print("Action space :", env.action_space)
    print("State space  :", env.observation_space)
    print("Dynamics parameters:", env.unwrapped.get_parameters())

    obs, _ = env.reset()
    total_reward = 0.0
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {step}, Action: {action}, Reward: {reward}")

        if args.render:
            env.render()

        if done or truncated:
            print(f"Episode finished after {step+1} steps. Total reward: {total_reward:.2f}")
            break

    env.close()

if __name__ == "__main__":
    main()
