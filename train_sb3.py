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

# WandbCallback permette di integrare facilmente wandb con gli algoritmi di RL di Stable Baselines3,
# automaticamente registrando dati come rewards, loss, e altri metrics durante l'addestramento.
import wandb
from wandb.integration.sb3 import WandbCallback

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

def make_model_name(algo: str, hypers: dict, timesteps: int,train_domain:str,test_domain:str,udr:str,counter: int = None) -> str:
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
    #parts.append(f"lr{hypers['learning_rate']:.0e}")
    #parts.append(f"bs{hypers['batch_size']}")
    if algo == "PPO":
        parts.append(f"{train_domain}")
        parts.append(f"{test_domain}")
        if udr:
            parts.append(f"udr")
        #parts.append(f"ns{hypers['n_steps']}")
    #parts.append(f"ts{timesteps//1000}k")  # es. ts100k, ts1000k
    #if counter is not None:
        #parts.append(f"run{counter}")
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
def create_callbacks(n_steps: int,
                     args, 
                     eval_env: gym.Env,
                     patience: int = 5,
                     min_evals: int = 10):
    run_name = f"{args.train_domain}_{args.test_domain}"
    if args.UDR:
        run_name += "_udr"
    best_model_path = f"./best_model/{run_name}"

    # Inserire 'checkpoint_callback nel return se si vuole utilizzare'
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_steps,
        save_path="./checkpoints/",
        name_prefix="rl_hopper"
    )
    # •	CheckpointCallback:
	#   •	Ogni save_freq di volte viene invocato env.step() chiama model.save(…)
	#   •	Dietro le quinte serializza: pesi reti, stato ottimizzatore, numero di passi.

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,  # save the best model with a unique name
        # log_path="./logs/",  # save the evaluations results in a NumPy archive
        eval_freq=50_000 // n_steps,
        deterministic=True,
        render=False
    )
    callbacks = [eval_callback]
    # •	EvalCallback:
	#   •	Ogni eval_freq batch esegue evaluate_policy(model, eval_env, …) internamente.
	#   •	Se la ricompensa media supera il best reward precedente, salva automaticamente con model.save(best_model_path)

    # stop_callback = StopTrainingOnNoModelImprovement(
    #     max_no_improvement_evals=patience,
    #     min_evals=min_evals,
    #     verbose=1
    # )
  #  stop_callback.parent = eval_callback

    # Callback di early stopping: interrompe se non migliora per 'patience' evals
    
    if args.WandDB:
    # Inizializzi i parametri di Wandb
        wandb.init(
            project="CustomHopper-RL",
            config={
                "algorithm": args.algo,
                "n_steps": n_steps,
                **COMMON_HYPERS,
                **ALG_HYPERS[args.algo],
            },
            name= f"{args.algo}-{args.train_domain}--{args.test_domain}", # personalizziamo il nome es. PPO-Source : Si potrebbe poi aggiungere per il Tuning nomi diversi
            sync_tensorboard=True, 
            monitor_gym=True,
        )
        
        # Callback per salvare modelli e loggare dati sul sito di Wandb
        wandb_callback = WandbCallback(
            model_save_freq=50_000,   # 50_000
            model_save_path=f"./wandb_models/{wandb.run.id}/",
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    return callbacks

def train_model(algo:str, hypers:dict,train_domain:str,test_domain:str, total_timesteps, callbacks, UDR:bool):
    if algo == "PPO":
        train_env = make_vec_env(lambda: make_env(train_domain, udr=UDR), n_envs=8)
    else:
        train_env = make_env(train_domain, udr=UDR)
    eval_env = make_env(test_domain, udr=False)

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
    start = time.time() # a che serve ?
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    end = time.time() # a che serve?
    
    if UDR:
        print("Training with Uniform Domain Randomization (UDR) enabled.")
    eval_env.reset(seed=42)
    mean_reward, std_reward = evaluate_policy(  # Calcola media e deviazione della somma di ricompense per episodio
        model,
        eval_env,
        n_eval_episodes=50,
        deterministic=True,  # scegli sempre l’azione più probabile (o la media) proposta dalla rete, eliminando la componente di casualità
        render=False,
        warn=False,  # # opzionale: sopprime warning su env non wrappers
    )

    return mean_reward, std_reward, model

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

     # create env
    eval_env = make_env(args.test_domain)
    train_env_for_eval = make_env(args.train_domain)

    # collect hyperparams
    hypers = {**COMMON_HYPERS, **ALG_HYPERS[algo]}
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.startswith(algo.lower()) and f.endswith(".zip")]
    counter = len(existing) + 1

    model_filename = make_model_name(algo, hypers, total_timesteps,args.train_domain,args.test_domain,args.UDR,counter)
   
    callbacks = create_callbacks(
    n_steps=hypers.get("n_steps", 1),
    args=args,
    eval_env=train_env_for_eval,
    patience=5,     # numero di valutazioni consecutive senza improvement
    min_evals=10    # numero minimo di valutazioni prima di iniziare a stoppare
)

#    print(f"Training time for {hypers['total_timesteps']} steps: {end - start:.2f} seconds")

    mean_reward, std_reward, model = train_model(algo, hypers, args.train_domain, 
                                                 args.test_domain, total_timesteps, callbacks, args.UDR)

    # save final
    model.save(os.path.join(save_dir, model_filename)) # salva pesi finali, ottimizzatori, parametri
    print(f"Final training ({algo}) on {args.train_domain} env: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Model saved as {model_filename}")

    # final eval
    # Otteniamo il Lower bound => performance media del modello source→target
    # Il risultato mean_reward è quindi la ricompensa media che ottieni appena trasferisci la policy “source”
    # sul dominio “target” senza alcun adattamento
    eval_env.reset(seed=42)
    mean_reward_eval, std_reward_eval = evaluate_policy(  # Calcola media e deviazione della somma di ricompense per episodio
        model,
        eval_env,
        n_eval_episodes=50,
        deterministic=True,  # scegli sempre l’azione più probabile (o la media) proposta dalla rete, eliminando la componente di casualità
        render=False,
        warn=False,  # # opzionale: sopprime warning su env non wrappers
    )
    print(f"Final evaluation ({algo}) on {args.target_domain} env: mean_reward={mean_reward_eval:.2f} +/- {std_reward_eval:.2f}")

    # rimuove completamente la cartella checkpoints/
    ckpt_dir = "./checkpoints"
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"Removed checkpoint directory: {ckpt_dir}")


if __name__ == "__main__":
    main()