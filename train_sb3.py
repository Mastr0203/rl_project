"""
Sample script for training a control policy on CustomHopper using
stable-baselines3 (≥ 2.0, compatibile con Gymnasium).

Leggi la documentazione SB3 e implementa (TASK 4-5) un training pipeline
con PPO o SAC. Questo file si limita a creare l'ambiente e a stampare
informazioni utili, lasciando i TODO dove richiesto.
"""

from __future__ import annotations

import time
import shutil
import os

import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC  # sono le classi di SB3 che incapsulano l’algoritmo (rete policy + ottimizzatori + loop di training)
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
# StopTrainingOnNoModelImprovement controlla la reward media di ogni eval (via EvalCallback) e conta quante volte
# consecutive non c’è miglioramento rispetto al best so far.
#   • Se supera patience valutazioni consecutive senza alcun miglioramento, interrompe il training.
#	• min_evals assicura che venga fatto almeno un numero minimo di valutazioni prima di poter fermare (es. 10).
# CheckpointCallback salva periodicamente pesi e ottimizzatori allo stato corrente
# EvalCallback esegue valutazioni regolari su un env di test e salva il “best model”
from stable_baselines3.common.evaluation import evaluate_policy  # è una funzione helper che, dato un modello SB3
                                                                 # addestrato, lo esegue per N episodi e restituisce
                                                                 # media e deviazione delle ricompense
from stable_baselines3.common.env_util import make_vec_env

# Import custom Hopper environments (register on import)
from env.custom_hopper import *  # noqa: F401,F403

def make_model_name(algo: str, hypers: dict, timesteps: int, counter: int = None) -> str:
    """
    Restituisce un nome di file basato su:
      - nome algoritmo (ppo / sac)
      - learning_rate
      - batch_size
      - n_steps (solo per PPO)
      - numero di timesteps di training
      - opzionale counter incrementale
    """
    parts = [ algo.lower() ]
    # i parametri che di solito modifichi
    parts.append(f"lr{hypers['learning_rate']:.0e}")
    parts.append(f"bs{hypers['batch_size']}")
    if algo == "PPO":
        parts.append(f"ns{hypers['n_steps']}")
    parts.append(f"ts{timesteps//1000}k")  # es. ts100k, ts1000k
    if counter is not None:
        parts.append(f"run{counter}")
    name = "_".join(parts) + ".zip"
    return name

# -----------------------------
# 1. Hyperparameters
# -----------------------------
COMMON_HYPERS = {
    "total_timesteps": 100_000,  # è il conteggio complessivo di passi‐ambiente (ossia di (state, action, reward))
    "gamma": 0.99,
    "tensorboard_log": "./tb",
}

ALG_HYPERS = {
    "PPO": {
        "learning_rate": 3e-4,  # testare 1e-4 e 5e-4
        "batch_size": 64,  # suddivisione del batch in minibatch per più epoch
        "n_steps": 2048,  # ogni quante interazioni con l’ambiente SB3 calcola un batch per il gradient-update
        "clip_range": 0.2,
        "ent_coef": 0.0,  # Entropy coefficient for the loss calculation
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
def create_callbacks(n_steps: int, 
                     eval_env: gym.Env,
                     patience: int = 5,
                     min_evals: int = 10):
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_steps,
        save_path="./checkpoints/",
        name_prefix="rl_hopper"
    )
    # •	CheckpointCallback:
	#   •	Ogni save_freq di volte viene invocato env.step() chiama model.save(…)
	#   •	Dietro le quinte serializza: pesi reti, stato ottimizzatore, numero di passi.
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=patience,
        min_evals=min_evals,
        verbose=1
    )
    # Callback di early stopping: interrompe se non migliora per 'patience' evals
    eval_env = make_env("target")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",  # save the best model
        # log_path="./logs/",  # save the evaluations results in a NumPy archive
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
    train_env = make_vec_env(lambda: make_env("source"), n_envs=8)  # accelera e stabilizza il learning passando
                                                                    # un vec_env con più ambienti in parallelo
    eval_env = make_env("target")

    # collect hyperparams
    hypers = {**COMMON_HYPERS, **ALG_HYPERS[algo]}
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.startswith(algo.lower()) and f.endswith(".zip")]
    counter = len(existing) + 1

    model_filename = make_model_name(algo, hypers, hypers['total_timesteps'], counter)
    # instantiate model
    if algo == "PPO":
        model = PPO(
            "MlpPolicy",  # policy network pre-definita, una MLP a 2 layer con attivazione Tanh
            train_env,  # The environment to learn from
            learning_rate=hypers['learning_rate'],
            gamma=hypers['gamma'],
            n_steps=hypers['n_steps'],
            batch_size=hypers['batch_size'],
            clip_range=hypers['clip_range'],
            ent_coef=hypers['ent_coef'],
            tensorboard_log=f"{hypers['tensorboard_log']}/{model_filename}",
            verbose=1,
        )
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
            tensorboard_log=f"{hypers['tensorboard_log']}/{model_filename}",
            verbose=1,
        )

    callbacks = create_callbacks(
    n_steps=hypers.get("n_steps", 1),
    eval_env=eval_env,
    patience=5,     # numero di valutazioni consecutive senza improvement
    min_evals=10    # numero minimo di valutazioni prima di iniziare a stoppare
)
    # TRAIN
    # 1. Ciclo principale:
	#   • Raccoglie n_steps interazioni da train_env.
	#   • Calcola advantage / target critic.
	#   • Esegue n_epochs di gradient descent sui minibatch interni (solo PPO).
	#   • Chiede ai callback di checkpoint/eval di agire.
	# 2.	Continua fino a total_timesteps.
    start = time.time()
    model.learn(
        total_timesteps=hypers['total_timesteps'],
        callback=callbacks
    )
    end = time.time()

    print(f"Training time for {hypers['total_timesteps']} steps: {end - start:.2f} seconds")

    # save final
    model.save(os.path.join(save_dir, model_filename)) # salva pesi finali, ottimizzatori, parametri
    print(f"Model saved as {model_filename}")

    # final eval
    # Otteniamo il Lower bound => performance media del modello source→target
    # Il risultato mean_reward è quindi la ricompensa media che ottieni appena trasferisci la policy “source”
    # sul dominio “target” senza alcun adattamento
    eval_env.reset(seed=42)
    mean_reward, std_reward = evaluate_policy(  # Calcola media e deviazione della somma di ricompense per episodio
        model,
        eval_env,
        n_eval_episodes=50,
        deterministic=True,  # scegli sempre l’azione più probabile (o la media) proposta dalla rete, eliminando la componente di casualità
        render=False,
        warn=False,  # # opzionale: sopprime warning su env non wrappers
    )
    print(f"Final evaluation ({algo}) on target env: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # rimuove completamente la cartella checkpoints/
    ckpt_dir = "./checkpoints"
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"Removed checkpoint directory: {ckpt_dir}")


if __name__ == "__main__":
    main()