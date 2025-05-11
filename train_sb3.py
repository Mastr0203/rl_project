"""
Sample script for training a control policy on CustomHopper using
stable-baselines3 (≥ 2.0, compatibile con Gymnasium).

Leggi la documentazione SB3 e implementa (TASK 4-5) un training pipeline
con PPO o SAC. Questo file si limita a creare l'ambiente e a stampare
informazioni utili, lasciando i TODO dove richiesto.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC  # sono le classi di SB3 che incapsulano l’algoritmo (rete policy + ottimizzatori + loop di training)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# CheckpointCallback salva periodicamente pesi e ottimizzatori allo stato corrente
# EvalCallback esegue valutazioni regolari su un env di test e salva il “best model”
from stable_baselines3.common.evaluation import evaluate_policy  # è una funzione helper che, dato un modello SB3
                                                                 # addestrato, lo esegue per N episodi e restituisce
                                                                 # media e deviazione delle ricompense

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
        "batch_size": 64,  # suddivisione del batch in minibatch per più epoch
        "n_steps": 2048,  # ogni quante interazioni con l’ambiente SB3 calcola un batch per il gradient-update
        "clip_range": 0.2,
        "ent_coef": 0.0,
    },
    "SAC": {
        "learning_rate": 3e-4,
        # SAC specific
        "batch_size": 256,
        "train_freq": 1,  # ogni quante azioni campiona un gradient-step
        "learning_starts": 10000,  # attendi tante azioni prima di iniziare ad allenare (riempimento buffer)
        "buffer_size": 1000000,  # dimensione del replay buffer
        "ent_coef": 'auto',  # SAC sceglie automaticamente il coefficiente di entropia, bilanciando esplorazione e sfruttamento
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
"""La funzione create_callbacks(n_steps) ha lo scopo di generare e restituire due callback da passare al metodo
    model.learn(), in modo da automatizzare:
	1.	CheckpointCallback
	    •	Cosa fa: salva periodicamente lo stato del modello (pesi, ottimizzatori, contatore di timesteps) su
            disco.
	    •	Perché è utile: se l'allenamento viene interrotto (per es. crash, timeout, riavvio), puoi ripartire
            da un ultimo salvataggio anziché ricominciare da zero. Inoltre ti permette di conservare più snapshot
            del modello lungo il training.
	2.	EvalCallback
	    •	Cosa fa: ogni tot passi di ambiente valuta il modello su un ambiente di test (qui CustomHopper-target-v0),
            calcolando la ricompensa media su alcuni episodi.
	    •	Perché è utile:
	        •	Ti dà un feedback regolare sulle reali prestazioni di transfer (sim-to-sim) man mano che l'agente
                impara.
	        •	Se ottiene un nuovo record di ricompensa media, salva quel “best model” in una cartella dedicata.
	        •	Registra metriche (reward, std, …) in un file di log per poterle visualizzare con TensorBoard o
                altri strumenti."""
def create_callbacks(n_steps: int):
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_steps,
        save_path="./checkpoints/",
        name_prefix="rl_hopper"
    )
    # •	CheckpointCallback:
	#   •	Ogni save_freq batch chiama model.save(…).
	#   •	Dietro le quinte serializza: pesi reti, stato ottimizzatore, numero di passi.
    eval_env = make_env("target")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=50_000 // n_steps,
        deterministic=True,
        render=False
    )
    # •	EvalCallback:
	#   •	Ogni eval_freq batch esegue evaluate_policy(model, eval_env, …) internamente.
	#   •	Se la ricompensa media supera il best reward precedente, salva automaticamente con model.save(best_model_path)
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
            "MlpPolicy",  # policy network pre-definita, una MLP a 2 layer con attivazione Tanh
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
    # 1. Ciclo principale:
	#   • Raccoglie n_steps interazioni da train_env.
	#   • Calcola advantage / target critic.
	#   • Esegue n_epochs di gradient descent sui minibatch interni (solo PPO).
	#   • Chiede ai callback di checkpoint/eval di agire.
	# 2.	Continua fino a total_timesteps.
    model.learn(
        total_timesteps=hypers['total_timesteps'],
        callback=callbacks
    )

    # save final
    model.save(f"{algo.lower()}_custom_hopper_final") # salva pesi finali, ottimizzatori, parametri

    # final eval
    mean_reward, std_reward = evaluate_policy(  # Calcola media e deviazione della somma di ricompense per episodio
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Final evaluation ({algo}) on target env: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()