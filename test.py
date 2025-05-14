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


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="path .pt/.pth file")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="torch device")
    p.add_argument("--render", action="store_true", help="visualizza la GUI")
    p.add_argument("--episodes", type=int, default=10, help="# episodi di test")
    return p.parse_args()


args = parse_args()


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> None:
    env = gym.make(
        "CustomHopper-source-v0",            # o "CustomHopper-target-v0"
        render_mode="human" if args.render else None,
    )

    print("Action space :", env.action_space)
    print("State space  :", env.observation_space)
    print("Dynamics parameters:", env.unwrapped.get_parameters())

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    max_action = env.action_space.high

    policy = Policy(obs_dim, act_dim)
    policy.load_state_dict(torch.load(args.model, map_location=args.device), strict=False) #True

    agent = Agent(policy, device=args.device, max_action=max_action)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        terminated = truncated = False
        ep_return = 0.0

        while not (terminated or truncated):
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward

            if args.render:
                env.render()  # opzionale; la finestra è già aperta con render_mode="human"

        print(f"Episode {ep:3d} | Return: {ep_return:.2f}")

    env.close()


if __name__ == "__main__":
    main()