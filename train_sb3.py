"""
Sample script for training a control policy on CustomHopper using
stable‑baselines3 (≥ 2.0, compatibile con Gymnasium).

Leggi la documentazione SB3 e implementa (TASK 4‑5) un training pipeline
con PPO o SAC. Questo file si limita a creare l'ambiente e a stampare
informazioni utili, lasciando i TODO dove richiesto.
"""

from __future__ import annotations

import gymnasium as gym

# importa CustomHopper-* (i register sono eseguiti all'import)
from env.custom_hopper import *  # noqa: F401,F403


def main() -> None:
    train_env = gym.make("CustomHopper-source-v0")  # render_mode=None (headless)

    print("State space :", train_env.observation_space)
    print("Action space:", train_env.action_space)
    print("Dynamics parameters:", train_env.get_parameters())

    # -----------------------------------------------------------------
    # TASK 4 & 5
    #   • Scegli un algoritmo (PPO o SAC)
    #   • Instanzia il modello da stable_baselines3
    #   • addestra sul train_env
    #   • valuta sul target oppure sullo stesso env
    #
    # Esempio (da completare / parametri a scelta):
    #
    # from stable_baselines3 import PPO
    # model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./tb")
    # model.learn(total_timesteps=1_000_000)
    # model.save("ppo_custom_hopper")
    # -----------------------------------------------------------------


if __name__ == "__main__":
    main()