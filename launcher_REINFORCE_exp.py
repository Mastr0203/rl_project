import subprocess
from multiprocessing import Process
import os

# Cartella madre in cui verranno create tutte le sottocartelle
default_parent = "REINFORCE_experiments"

# Percorso completo a train.py (assume sia nella stessa cartella di questo launcher)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")

# Definisci i quattro set di iperparametri
experiments = [
    {"gamma": 0.95,  "hidden": 64,  "lr_policy": 1e-4, "lr_critic": 5e-4, "baseline": "20.0"},
    {"gamma": 0.95,  "hidden": 128, "lr_policy": 1.0,  "lr_critic": 1e-4, "baseline": "20.0"},
    {"gamma": 0.99,  "hidden": 128, "lr_policy": 1.0,  "lr_critic": 1e-4, "baseline": "0.0"},
    {"gamma": 0.995, "hidden": 32,  "lr_policy": 1.0,  "lr_critic": 5e-4, "baseline": "dynamic"},
]

# Crea la cartella madre se non esiste
os.makedirs("models/"+default_parent, exist_ok=True)


def run_exp(params):
    # Nome univoco per ogni esperimento
    name = (f"exp_gamma{params['gamma']}_h{params['hidden']}"
            f"_lp{params['lr_policy']}_lc{params['lr_critic']}"
            f"_b{params['baseline']}")

    # Path completo sotto la cartella madre
    dir_path = os.path.join(default_parent, name)
    os.makedirs(dir_path, exist_ok=True)

    # Imposta le variabili d'ambiente per WandB
    env = os.environ.copy()
    env["WANDB_MODE"] = "online"  # salva sul cloud

    # Costruisci comando con percorso assoluto a train.py
    cmd = [
        "python", TRAIN_SCRIPT,
        "--n-episodes=200000",
        f"--gamma={params['gamma']}",
        f"--hidden={params['hidden']}",
        f"--lr-policy={params['lr_policy']}",
        f"--lr-critic={params['lr_critic']}",
        f"--baseline={params['baseline']}",
        "--WandDB",
        "--domain=source",
        "--device=cpu",
    ]

    print(f"[{name}] Launching in {dir_path}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=dir_path, check=True, env=env)


if __name__ == "__main__":
    procs = []
    for params in experiments:
        p = Process(target=run_exp, args=(params,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("Tutti gli esperimenti sono terminati.")
