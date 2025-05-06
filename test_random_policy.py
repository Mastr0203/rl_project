"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                - What is the action space in the Hopper environment? Is it discrete or continuous?
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                - When exactly is the episode over?
                - What is an action here?
"""
import gymnasium as gym
import time

# L'import serve solo per registrare CustomHopper-* tramite le
# chiamate `register()` presenti in env/custom_hopper.py
from env.custom_hopper import *  # noqa: F401,F403

def main() -> None:
    render = True
    env = gym.make(
        "CustomHopper-source-v0",          # ["CustomHopper-target-v0", "CustomHopper-source-v0"]
        render_mode="human" if render else None,
    )
    print("Ambiente registrato correttamente.")
    
    #env.model.body_mass[1] = 5.0  # Cambia la massa del torso
    #env.model.geom_friction[0] = [1.0, 0.5, 0.5]  # Cambia l'attrito

    print("State space :", env.observation_space)  # È un Box, quindi continuo.
	                                               #  •	La dimensione (11,) → vettore con 11 componenti reali
	                                               #  •	Contiene posizione e velocità (in parte)
                                                   #  => Conclusione: state space = continuo
    print("Action space:", env.action_space)       #  È un Box → azione continua
	                                               #  •	Ha 3 valori: torques da applicare ai 3 giunti principali
                                                   #  => Conclusione: action space = continuo
    print("Dynamics parameters:", env.unwrapped.get_parameters())  # masse dei link
    # gym.pprint_registry()  #  Pretty prints all environments in the registry

    n_episodes = 10

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42)  # riporta l’ambiente al suo stato iniziale e inizia un nuovo episodio
                                 		# 	• observation: l’osservazione iniziale → un array NumPy che rappresenta lo stato iniziale (es: posizione e velocità dei giunti).
	                             		#  • info: un dizionario opzionale con informazioni aggiuntive (es: configurazioni particolari dell’ambiente, dettagli utili).
        terminated = truncated = False  # Flag che indica se l’episodio è terminato
        n_steps = 0   # Contatore del numero di step per episodio
        
        while not (terminated or truncated):
            action = env.action_space.sample()  # Genera un'azione casuale valida dallo spazio delle azioni
            obs, reward, terminated, truncated, info = env.step(action)
			# Esegue l'azione e ottiene:
            # - obs: nuovo stato
            # - reward: ricompensa ottenuta
            # - terminated: True se l’agente ha fallito (es: è caduto)
            # - truncated: True se si è raggiunto il limite massimo di tempo
            # - info: dizionario con eventuali dettagli extra

            if render:
                env.render()               # facoltativo: la finestra è già aperta
            
            print(f"[Ep {ep:2d} | Step {n_steps:3d}] Reward: {reward:.3f}")
            n_steps += 1
        
		# Analizza tutte le cause di fine episodio
        if terminated and not truncated:
            print("⚠️ Episodio terminato perché Hopper è diventato unhealthy (caduto o sbilanciato).")
        if truncated:
            print("⏳ Truncated!")
            if info.get("time_limit_reached", False):
                print(" - ⏰ Tempo massimo (1000 timestep) raggiunto.")
            else:
                print(" - ℹ️ Troncato ma motivo non specificato nei campi info.")
        # Dopo l’ultimo episodio, lascia la finestra aperta
        if ep == n_episodes - 1:
            print("✅ Ultimo episodio completato. La finestra resterà aperta.")
            print("👉 Premi CTRL+C nel terminale per uscire.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                break

    env.close()  # Chiude l’ambiente e libera le risorse


if __name__ == "__main__":
    main()
# Esegue la funzione main() solo se lo script viene eseguito direttamente