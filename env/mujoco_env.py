"""
env/mujoco_env.py
-----------------

Compatibilità post‑migrazione a Gymnasium (>= 1.0).

Il vecchio wrapper basato su `mujoco_py` non è più necessario: re‑esportiamo
semplicemente la classe `MujocoEnv` di Gymnasium, così istruzioni come

    from env.mujoco_env import MujocoEnv

continuano a funzionare senza modifiche altrove.

Se in futuro ti dovesse servire aggiungere logica custom comune a più env,
puoi re‑introdurre metodi mix‑in su una nuova sottoclasse, ma per ora non serve.
"""

from gymnasium.envs.mujoco import MujocoEnv  # noqa: F401

__all__ = ["MujocoEnv"]