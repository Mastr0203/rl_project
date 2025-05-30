from __future__ import annotations

"""Test an RL agent on the OpenAI Gym Hopper environment"""
"""
test.py
-------
Valuta una policy salvata su CustomHopper-{source|target}-v0
dopo il refactor a Gymnasium (>= 1.0).
"""



import argparse
from pathlib import Path

import torch
import gymnasium as gym

# registra gli env CustomHopper-*
from env.custom_hopper import *  # noqa: F401,F403
from agent import Agent, Policy
from stable_baselines3 import PPO, SAC


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="path .pt/.pth file")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="torch device")
    p.add_argument("--render", action="store_true", help="visualizza la GUI")
    p.add_argument("--episodes", type=int, default=10, help="# episodi di test")
    p.add_argument("--dimension", default="target", choices=["source", "target"], help="Domain to test on [source, target]")
    p.add_argument("--algo", default="PPO", choices=["REINFORCE", "ACTORCRITIC", "PPO", "SAC"], help="Algorithm to use")
    return p.parse_args()


args = parse_args()


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> None:
    env = gym.make(
        f"CustomHopper-{args.dimension}-v0",            # o "CustomHopper-target-v0"
        render_mode="human" if args.render else None,
    )

    print("Action space :", env.action_space)
    print("State space  :", env.observation_space)
    print("Dynamics parameters:", env.unwrapped.get_parameters())

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    max_action = env.action_space.high
    algo_basic = args.algo == "REINFORCE" or args.algo== "ACTORCRITIC"

    if args.algo == "REINFORCE" or args.algo== "ACTORCRITIC":
        policy = Policy(obs_dim, act_dim)
        policy.load_state_dict(torch.load(args.model, map_location=args.device), strict=False) #True

        agent = Agent(policy, device=args.device, max_action=max_action) # agent al posto di model
    elif args.algo == "PPO":
        model = PPO.load(args.model)
    elif args.algo == "SAC":
        model = SAC.load(args.model)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        terminated = truncated = False
        ep_return = 0.0

        while not (terminated or truncated):
            if algo_basic:
                action, _ = agent.get_action(obs, evaluation=True)
            else:
                action, _= model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward

            if args.render:
                env.render()  # opzionale; la finestra è già aperta con render_mode="human"

        print(f"Episode {ep:3d} | Return: {ep_return:.2f}")

    env.close()


if __name__ == "__main__":
    main()