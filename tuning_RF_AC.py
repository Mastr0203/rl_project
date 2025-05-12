import os
import csv
import itertools
import torch
import gymnasium as gym

from env.custom_hopper import *  # registra gli env CustomHopper-*
from agent import Agent, Policy, Critic

# --- HYPERPARAMETER GRID ---
gammas = [0.95, 0.99, 0.995]
hiddens = [32, 64, 128]
lrs = [1e-4, 5e-4, 1e-3]

# Numero di episodi per run (puoi abbassarlo in fase di test)
N_EPISODES = 20_000
PRINT_EVERY = 2_000
ALG = "ActorCritic"  # o "REINFORCE"

# Salva risultati
os.makedirs("tuning_results", exist_ok=True)
with open("tuning_results/summary_{ALG}.csv", "w", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=[
        "gamma", "hidden", "lr_policy", "lr_critic", "final_return"
    ])
    writer.writeheader()

    # Loop sulle combinazioni
    for gamma, hidden, lr in itertools.product(gammas, hiddens, lrs):
        # Setup env
        env = gym.make("CustomHopper-source-v0")
        obs_dim = env.observation_space.shape[-1]
        act_dim = env.action_space.shape[-1]
        max_action = env.action_space.high

        # Istanzia policy, critic e agent con iperparametri
        policy = Policy(obs_dim, act_dim, hidden=hidden)
        critic = Critic(obs_dim, act_dim, hidden=hidden) if ALG=="ActorCritic" else None
        agent = Agent(
            policy=policy,
            critic=critic,
            max_action=max_action,
            device="cpu",
            gamma=gamma,
            lr_policy=lr,
            lr_critic=lr
        )

        # Training loop
        last_return = None
        for ep in range(1, N_EPISODES+1):
            obs, _ = env.reset(seed=ep)
            done = False
            ep_ret = 0.0
            while not done:
                action, logp = agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_outcome(obs, next_obs, logp, reward, done)
                obs = next_obs
                ep_ret += reward

            agent.update_policy(ALG)
            last_return = ep_ret

            if ep % PRINT_EVERY == 0:
                print(f"[γ={gamma:.3f} h={hidden} lr={lr:.1e}] Ep {ep} → return {ep_ret:.2f}")

        # Scrivi nel CSV
        writer.writerow({
            "gamma": gamma,
            "hidden": hidden,
            "lr_policy": lr,
            "lr_critic": lr,
            "final_return": last_return
        })

        env.close()

print("Tuning completato. Risultati in tuning_results/summary.csv")