"""
Sample script for training a control policy on CustomHopper using
stable-baselines3 (≥ 2.0, compatibile con Gymnasium).

Leggi la documentazione SB3 e implementa (TASK 4-5) un training pipeline
con PPO o SAC. Questo file si limita a creare l'ambiente e a stampare
informazioni utili, lasciando i TODO dove richiesto.
"""

from __future__ import annotations

import time
import os

import wandb

import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC 
from callbacks import EvalCallback

from stable_baselines3.common.evaluation import evaluate_policy  
from stable_baselines3.common.env_util import make_vec_env

# Import custom Hopper environments (register on import)
from env.custom_hopper import *  # noqa: F401,F403

def make_model_name(algo: str,train_domain:str,test_domain:str,udr:str,counter: int = None) -> str:
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
    parts.append(f"{train_domain}")
    parts.append(f"{test_domain}")
    if udr:
        parts.append(f"udr")
    if counter != None:
        parts.append(str(counter))
    name = "_".join(parts) + ".zip"
    
    return name

# -----------------------------
# 1. Hyperparameters
# -----------------------------
total_timesteps = 1_000_000  # è il conteggio complessivo di passi‐ambiente (ossia di (state, action, reward))
                            #Ho settato 200_000 per tuning, mettere 1_000_000 per training

COMMON_HYPERS = {
     "gamma": 0.99,
     "tensorboard_log": "./tb",
}

ALG_HYPERS = {
    "PPO": {
        "learning_rate": 5e-4,  # testare 1e-4 e 5e-4
        "batch_size": 128,  # suddivisione del batch in minibatch per più epoch
        "n_steps": 1024,  # ogni quante interazioni con l’ambiente SB3 calcola un batch per il gradient-update
        "clip_range": 0.2,
        "ent_coef": 0.05,  # Entropy coefficient for the loss calculation    0.05
    },
    "SAC": {
        "learning_rate": 3e-4,              # Può essere tunato: 1e-4, 5e-4, 1e-3
        "batch_size": 256,                  # Tipici valori: 64, 128, 512
        "train_freq": 1,                    # Ogni quante azioni aggiornare
        "gradient_steps": 4,                # Quanti gradient step per aggiornamento
        "learning_starts": 5000,          # Quando iniziare il training
        "buffer_size": int(1e5),           # Dimensione del replay buffer
        "ent_coef": "auto",                 # Coefficiente di entropia (auto-tuning)
        "policy_kwargs": dict(net_arch=[256, 256]),  # Architettura della policy network
    }
}

# -----------------------------
# 2. Environment setup
# -----------------------------
def make_env(domain: str, udr: bool = False) -> gym.Env:
    env_id = f"CustomHopper-{domain}-v0"
    env = gym.make(env_id)
    if hasattr(env.unwrapped, "udr_enabled"):       #hasattr is used to call the argument "udr" with getatt
        env.unwrapped.udr_enabled = udr             #in the custom_hopper
    
    return env

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
def create_callbacks(n_steps: int,algo,test_domain,Wandb,name):
    best_model_path = f"./best_model"
    eval_env = make_env(test_domain)
    eval_env.reset(seed=42)

    eval_callback = EvalCallback(
        eval_env = eval_env,
        best_model_save_path=best_model_path,
        name = name,
        eval_freq=50_000 // n_steps,
        deterministic=True,
        render=False
    )
    callbacks = eval_callback
   
    if Wandb:
    # Inizializzi i parametri di Wandb
        wandb.init(
            project="CustomHopper-RL",
            config={
                "algorithm": algo,
                "n_steps": n_steps,
                **COMMON_HYPERS,
                **ALG_HYPERS[algo],
            },
            name= name,
            sync_tensorboard=True, 
            monitor_gym=True,
        )
    
    return callbacks


def train_model(algo:str, hypers:dict,train_domain:str,test_domain:str, total_timesteps, UDR:bool, WandDB:str,filename='T'):
    if algo == "PPO":
        train_env = make_vec_env(lambda: make_env(train_domain, udr=UDR), n_envs=8)
    else:
        train_env = make_env(train_domain, udr=UDR)
    eval_env = make_env(test_domain, udr=False)
    name = filename.strip(".zip")

    # TRAIN
    # 1. Ciclo principale:
	#   • Raccoglie n_steps interazioni da train_env.
	#   • Calcola advantage / target critic.
	#   • Esegue n_epochs di gradient descent sui minibatch interni (solo PPO).
	#   • Chiede ai callback di checkpoint/eval di agire.
	# 2.	Continua fino a total_timesteps.

    if algo == "PPO":
        model = PPO(
            "MlpPolicy",  # policy network pre-definita, una MLP a 2 layer con attivazione Tanh
            train_env,  # The environment to learn from
            **hypers
        )
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            train_env,
            **hypers
        )

    callbacks = create_callbacks(
        n_steps=hypers.get("n_steps", 1),
        algo = algo,
        test_domain = test_domain,
        Wandb = WandDB,
        name = name
        )

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    end = time.time()
    
    if UDR:
        print("Training with Uniform Domain Randomization (UDR) enabled.")
    best_model = PPO.load(filename)

    eval_env.reset(seed=42)
    mean_reward, std_reward = evaluate_policy(
        best_model,
        eval_env,
        n_eval_episodes=50,  #Tuning set 20
        deterministic=True,  # scegli sempre l’azione più probabile (o la media) proposta dalla rete, eliminando la componente di casualità
        render=False,
        warn=False,          #Optional
    )

    print(f"Training time: {end - start:.2f} seconds")

    return mean_reward, std_reward

# -----------------------------
# 4. Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PPO or SAC on CustomHopper")
    parser.add_argument("--algo", choices=["PPO", "SAC"], default="PPO",
                        help="Algorithm to use: PPO or SAC")
    parser.add_argument("--train_domain", choices=["source", "target"], default="source", help="Domain to train on [source, target]")
    parser.add_argument("--test_domain", choices=["source", "target"], default="target", help="Domain to test on [source, target]")
    parser.add_argument("--WandDB", action="store_true", help="Use WandDB Callback")
    parser.add_argument("--UDR", action="store_true", default=False)
    
    args = parser.parse_args()
    algo = args.algo

    # collect hyperparams
    hypers = {**COMMON_HYPERS, **ALG_HYPERS[algo]}
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.startswith(algo.lower()) and f.endswith(".zip")]
    counter = len(existing) + 1

    model_filename = make_model_name(algo,args.train_domain,args.test_domain,args.UDR,counter)
    

    mean_reward, std_reward = train_model(algo, hypers, args.train_domain, 
                                            args.test_domain, total_timesteps, args.UDR, args.WandDB,model_filename)

    # save final
    print(f"Final training ({algo}) on {args.train_domain} env: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Model saved as {model_filename}")


if __name__ == "__main__":
    main()