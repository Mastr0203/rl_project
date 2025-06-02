"""
Train an RL agent on CustomHopper using
    • REINFORCE  (Task 2)
    • Actor‑Critic (Task 3)

Compatibile con Gymnasium (>= 1.0) e torch.
"""

from __future__ import annotations

import argparse
import time

import wandb
import torch
import gymnasium as gym
import numpy as np
import os

# registra gli env CustomHopper-* con le chiamate register()
from env.custom_hopper import *  # noqa: F401,F403
from agent import Agent, Policy,Critic
from gymnasium.vector import AsyncVectorEnv


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=100_000, help="# episodi di training")
    p.add_argument("--print-every", type=int, default=20_000, help="log ogni N episodi")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="torch device")
    p.add_argument('--n-envs', default=1, type=int, help='Select number of training envs')
    p.add_argument("--render", action="store_true", help="visualizza la GUI durante il training")

    p.add_argument('--algorithm', default='REINFORCE', type=str, choices=['REINFORCE', 'ActorCritic'], help='Select the Model [REINFORCE, ActorCritic]')
    p.add_argument('--domain', default='source', choices=["source", "target"], help="Domain to train on [source, target]")
    p.add_argument("--WandDB", action="store_true", help="Use WandDB Callback")
    p.add_argument("--save", action="store_true", help="Save the model")

    p.add_argument('--baseline', default='0', type=str, help="Insert a value or write 'dynamic' to state-dependent baseline")
    p.add_argument('--AC-critic', default = 'Q', type=str, choices = ['Q', 'V'], help="Whether critic estimates Q(s,a) or V(s)")

    p.add_argument("--gamma", default=0.99, type = float, help="Gamma to discount future rewards")
    p.add_argument("--lr-policy", default=5e-4, type = float, help="Learning Rate for Policy Updates")
    p.add_argument("--lr-critic", default=5e-4, type = float, help="Learning Rate for Critic Updates")
    p.add_argument("--hidden", default=64, type=int, help='Hidden Layers for Critic and Actor Nets')

    return p.parse_args()


args = parse_args()


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> None:

    if args.WandDB:
         # Inizializza wandb
        wandb.init(
            project= "Reinforcement Learning",
            config={
                "algorithm": args.algorithm,
                "domain": args.domain,
                "n_episodes": args.n_episodes,
                "device": args.device,
            },
            name= f"{args.algorithm}-{args.domain}", # personalizziamo il nome es. Reinforce-Source : Si potrebbe poi aggiungere per il Tuning nomi diversi
            sync_tensorboard=True,
            monitor_gym=True,
        )

    def make_env(seed_offset):
        def _init():
            env = gym.make(f"CustomHopper-{args.domain}-v0")  # oppure "CustomHopper-target-v0"
            env.reset(seed=seed_offset)
            return env
        return _init

    env = AsyncVectorEnv([make_env(i) for i in range(args.n_envs)])

    def save(policy, critic, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        counter = len(os.listdir(dir_path)) + 1
        model_path = os.path.join(dir_path, f"model_{counter}_{args.domain}")
        os.makedirs(model_path)
        policy_path = os.path.join(model_path, f"policy.mdl")
        torch.save(policy.state_dict(), policy_path)
        if critic is not None:
            critic_path = os.path.join(model_path, f"critic.mdl")
            torch.save(critic.state_dict(), critic_path)

    print("Action space :", env.action_space)
    print("State space  :", env.observation_space)
    # print("Dynamics parameters:", env.unwrapped.get_parameters())

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    max_action = env.action_space.high

    args_policy = {
        "state_space" : obs_dim,
        "action_space" : act_dim,
        "hidden" : args.hidden
        }
    
    args_critic = {
        "state_space" : obs_dim,
        "action_space" : act_dim,
        "hidden" : args.hidden
    }
    
    args_agent = {
        "model" : args.algorithm,
        "max_action" : max_action,
        "device" : args.device,
        "gamma" : args.gamma,
        "lr_policy" : args.lr_policy,
        "lr_critic" : args.lr_critic,
        "baseline" : args.baseline,
        "AC_critic" : args.AC_critic
    }

    def REINFORCE():
        args_agent["policy"] = Policy(**args_policy)

        if args.baseline == "dynamic":
            args_critic["action_space"] = 0
            args_agent["critic"] = Critic(**args_critic)
            args_agent["baseline"] = 0
        else:
            args_agent["critic"] = None
            args_agent["baseline"] = float(args.baseline)
        
        agent = Agent(**args_agent)

        best_score = env.reward_range[0]
        score_history = []

        start_time = time.time()
        for episode in range(args.n_episodes):
            obs, _ = env.reset(seed=episode)
            terminated = [False] * env.num_envs
            ep_returns = np.zeros(env.num_envs)

            while not all(terminated):
                actions, log_probs = agent.get_action(obs)  # log_probs e actions sono vettori
                prev_obs = obs

                obs, rewards, term, trunc, _ = env.step(actions)
                done = np.logical_or(term, trunc)

                for i in range(env.num_envs):
                    if not terminated[i]:  # evita di accumulare dati per env già terminati
                        agent.store_outcome(prev_obs[i], obs[i], log_probs[i], rewards[i], done[i])
                        ep_returns[i] += rewards[i]
                terminated = np.logical_or(terminated, done)

            loss = agent.update_policy()

            score_history.append(np.mean(ep_returns))
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if args.save:
                    dir_path = 'models/REINFORCE'
                    save(agent.policy, agent.critic, dir_path)

              # Va fuori perchè deve aggiornare ad ogni episodio

            if args.WandDB:
            # Logging per WandB
                wandb.log({
                    "episode": episode + 1,
                    "mean_return": ep_returns.mean(),
                    "max_return": ep_returns.max(),
                    "min_return": ep_returns.min(),
                    "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                    "time_elapsed": time.time() - start_time,
                })

            if args.render:
                env.render()

            if (episode + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                print(f"[Episode {episode+1}] return = {ep_returns.mean():.2f} loss = {loss:.4f} elapsed = {elapsed:.1f}s")
                start_time = time.time()  # resetta il timer

        env.close()

        print(f"FINAL_RESULT: {ep_returns.mean():.2f}")

    def ACTORCRITIC():
        if args.AC_critic == 'V': # critic estimates the value function
            args_critic["action_space"] = 0 

        args_agent["policy"] = Policy(**args_policy)
        args_agent["critic"] = Critic(**args_critic)
        agent = Agent(**args_agent)

        best_score = env.reward_range[0]
        score_history = []

        start_time = time.time()
        for episode in range(args.n_episodes):
            # Reset vectorized env for this episode
            obs, _ = env.reset(seed=episode)
            terminated = [False] * env.num_envs
            ep_returns = np.zeros(env.num_envs)

            # Step until all environments are done
            while not all(terminated):
                actions, log_probs = agent.get_action(obs)
                prev_obs = obs

                obs, rewards, term, trunc, _ = env.step(actions)
                done = np.logical_or(term, trunc)

                for i in range(env.num_envs):
                    if not terminated[i]:
                        agent.store_outcome(prev_obs[i], obs[i], log_probs[i], rewards[i], done[i])
                        ep_returns[i] += rewards[i]
                terminated = np.logical_or(terminated, done)

                # Perform update at each step
                loss = agent.update_policy()
            
            score_history.append(np.mean(ep_returns))
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if args.save:
                    dir_path = 'models/ActorCritic'
                    save(agent.policy, agent.critic, dir_path)


            # Optional rendering
            if args.render:
                env.render()

            # Logging printout
            if (episode + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                print(f"[Episode {episode+1}] mean_return = {ep_returns.mean():.2f} loss = {loss:.4f} elapsed = {elapsed:.1f}s")
                start_time = time.time()

        # salva i pesi a fine training
        if args.save:
            dir_path = 'models/ActorCritic'
            os.makedirs(dir_path, exist_ok=True)
            counter = len(os.listdir(dir_path)) + 1
            model_file = f"policy_{args.domain}_run_{counter}.mdl"
            path = os.path.join([dir_path, model_file])
            torch.save(agent.policy.state_dict(), path)
            # salva il critic (se presente)
            model_critic = f"critic_{args.domain}_run_{counter}.mdl"
            path = os.path.join([dir_path, model_critic])
            torch.save(agent.critic.state_dict(), path)


        env.close()

        # Print final average return across all parallel envs
        print(f"FINAL_RESULT: {ep_returns.mean():.2f}")

    if args.algorithm == "REINFORCE":
        REINFORCE()
    elif args.algorithm == "ActorCritic":
        ACTORCRITIC()

if __name__ == "__main__":
    main()
