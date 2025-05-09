"""
Sample script for training a control policy on CustomHopper using
stable‑baselines3 (≥ 2.0, compatibile con Gymnasium).

Leggi la documentazione SB3 e implementa (TASK 4‑5) un training pipeline
con PPO o SAC. Questo file si limita a creare l'ambiente e a stampare
informazioni utili, lasciando i TODO dove richiesto.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Import custom Hopper environments (register on import)
from env.custom_hopper import *  # noqa: F401,F403

# -----------------------------
# 1. Hyperparameters
# -----------------------------
COMMON_HYPERS = {
    "total_timesteps": 1_000_000,
    "gamma": 0.99,
    "tensorboard_log": "./tb",
}

ALG_HYPERS = {
    "PPO": {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 2048,
        "clip_range": 0.2,
        "ent_coef": 0.0,
    },
    "SAC": {
        "learning_rate": 3e-4,
        # SAC specific
        "batch_size": 256,
        "train_freq": 1,
        "learning_starts": 10000,
        "buffer_size": 1000000,
        "ent_coef": 'auto',
    }
}

# -----------------------------
# 2. Environment setup
# -----------------------------
def make_env(domain: str) -> gym.Env:
    env_id = f"CustomHopper-{domain}-v0"
    return gym.make(env_id)

# -----------------------------
# 3. Callbacks
# -----------------------------
def create_callbacks(n_steps: int):
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_steps,
        save_path="./checkpoints/",
        name_prefix="rl_hopper"
    )
    eval_env = make_env("target")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=50_000 // n_steps,
        deterministic=True,
        render=False
    )
    return [checkpoint_callback, eval_callback]

# -----------------------------
# 4. Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PPO or SAC on CustomHopper")
    parser.add_argument("--algo", choices=["PPO", "SAC"], default="PPO",
                        help="Algorithm to use: PPO or SAC")
    args = parser.parse_args()
    algo = args.algo

    # create envs
    train_env = make_env("source")
    eval_env = make_env("target")

    # collect hyperparams
    hypers = {**COMMON_HYPERS, **ALG_HYPERS[algo]}
    # instantiate model
    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=hypers['learning_rate'],
            gamma=hypers['gamma'],
            n_steps=hypers['n_steps'],
            batch_size=hypers['batch_size'],
            clip_range=hypers['clip_range'],
            ent_coef=hypers['ent_coef'],
            tensorboard_log=f"{hypers['tensorboard_log']}/{algo.lower()}_hopper",
            verbose=1,
        )
        callbacks = create_callbacks(hypers['n_steps'])
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=hypers['learning_rate'],
            gamma=hypers['gamma'],
            batch_size=hypers['batch_size'],
            train_freq=hypers['train_freq'],
            learning_starts=hypers['learning_starts'],
            buffer_size=hypers['buffer_size'],
            ent_coef=hypers['ent_coef'],
            tensorboard_log=f"{hypers['tensorboard_log']}/{algo.lower()}_hopper",
            verbose=1,
        )
        callbacks = create_callbacks(hypers.get('n_steps', 1))

    # train
    model.learn(
        total_timesteps=hypers['total_timesteps'],
        callback=callbacks
    )

    # save final
    model.save(f"{algo.lower()}_custom_hopper_final")

    # final eval
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Final evaluation ({algo}) on target env: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()