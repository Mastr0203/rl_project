"""
Train an RL agent on CustomHopper using
    • REINFORCE  (Task 2)
    • Actor‑Critic (Task 3)

Compatibile con Gymnasium (>= 1.0) e torch.
"""

from __future__ import annotations

import argparse

import torch
import gymnasium as gym

# registra gli env CustomHopper-* con le chiamate register()
from env.custom_hopper import *  # noqa: F401,F403
from agent import Agent, Policy,Critic


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=100_000, help="# episodi di training")
    p.add_argument("--print-every", type=int, default=20_000, help="log ogni N episodi")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="torch device")
    p.add_argument("--render", action="store_true", help="visualizza la GUI durante il training")

    p.add_argument('--algorithm', default='REINFORCE', type=str, help='Selected Model [REINFORCE, ActorCritic]')

    return p.parse_args()


args = parse_args()


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> None:
    env = gym.make(
        "CustomHopper-source-v0",             # oppure "CustomHopper-target-v0"
        render_mode="human" if args.render else None,
    )

    print("Action space :", env.action_space)
    print("State space  :", env.observation_space)
    print("Dynamics parameters:", env.unwrapped.get_parameters())

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    #policy = Policy(obs_dim, act_dim)
    #agent = Agent(policy, device=args.device)

    def REINFORCE():
        observation_space_dim = env.observation_space.shape[-1]
        action_space_dim = env.action_space.shape[-1]
        policy = Policy(observation_space_dim, action_space_dim)
        agent = Agent(policy, device=args.device)

        for episode in range(args.n_episodes):
            obs, _ = env.reset(seed=episode)
            terminated = truncated = False
            ep_return = 0.0

            while not (terminated or truncated):
                action, log_prob = agent.get_action(obs)
                prev_obs = obs

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_outcome(prev_obs, obs, log_prob, reward, done)
                ep_return += reward

            loss = agent.update_policy(args.algorithm)              #Va fuori perchè deve aggiornare ad ogni episodio

            if args.render:
                env.render()

            if (episode + 1) % args.print_every == 0:
                print(f"[Episode {episode+1}] return = {ep_return:.2f} loss = {loss}")

        # salva i pesi a fine training
        torch.save(agent.policy.state_dict(), "model.mdl")
        env.close()

    def ACTORCRITIC():
        observation_space_dim = env.observation_space.shape[-1]
        action_space_dim = env.action_space.shape[-1]

        policy = Policy(observation_space_dim, action_space_dim)
        critic = Critic(observation_space_dim, action_space_dim)
        agent = Agent(policy, device=args.device,critic=critic)
        
        for episode in range(args.n_episodes):
            obs, _ = env.reset(seed=episode)
            terminated = truncated = False
            ep_return = 0.0

            while not (terminated or truncated):
                action, log_prob = agent.get_action(obs)
                prev_obs = obs

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_outcome(prev_obs, obs, log_prob, reward, done)
                ep_return += reward

                loss = agent.update_policy(args.algorithm)      #Aggiorna ad ogni step

                if args.render:
                    env.render()

            if (episode + 1) % args.print_every == 0:
                print(f"[Episode {episode+1}] return = {ep_return:.2f} loss = {loss}")

        # salva i pesi a fine training
        torch.save(agent.policy.state_dict(), "model.mdl")
        env.close()
    

    if args.algorithm == "REINFORCE":
        REINFORCE()
    elif args.algorithm == "ActorCritic":
        ACTORCRITIC()

if __name__ == "__main__":
    main()
