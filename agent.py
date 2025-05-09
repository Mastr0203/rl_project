import numpy as np
import torch
import torch.nn.functional as F               # <‑‑ import usato solo nel critic loss
from torch.distributions import Normal

# Scopo: calcolare i ritorni scontati su tutta la traiettoria
def discount_rewards(r, gamma):  # r: tensor 1-D di ricompense raccolte in un episodio
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r  # Output: tensor discounted_r della stessa forma di r, contenente i ritorni scontati.


class Critic(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_critic = torch.nn.Linear(state_space + action_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_mean = torch.nn.Linear(self.hidden, 1)
        self.init_weights()

    # Il metodo restituisce un’unica predizione scalare per ciascuna coppia (s,a)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Input: stato s e azione a, concatenati in un vettore
                                                # di dimensione state_space + action_space
        x = self.tanh(self.fc1_critic(x))
        x = self.tanh(self.fc2_critic(x))
        return self.fc3_critic_mean(x)  # Output: Linear(hidden → 1) restituisce la stima del valore Q(s,a)
                                        # -> Rappresenta la stima del ritorno cumulato scontato (cumulative
                                        #   discounted reward) che ci si aspetta di raccogliere a partire
                                        #   dallo stato s, eseguendo l’azione a e poi proseguendo secondo
                                        #   la policy corrente.

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):  # il loop serve a individuare ogni torch.nn.Linear presente
                                                # nella rete e a inizializzarne i pesi e i bias secondo lo schema
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

# In un algoritmo di policy gradient continuo (come REINFORCE o Actor-Critic), la policy pi(a | s) viene
# spesso modellata come una distribuzione di probabilità parametrica, tipicamente una Gaussiana multivariata.
# Per definire completamente una Gaussiana servono due vettori di parametri:
#	•	la media
#	•	la deviazione standard
# Output: una distribuzione da cui l’Agent può campionare azioni esplorative e calcolare
#         log-prob per la loss del policy gradient.
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        # Il terzo layer un vettore di dimensione pari al numero di gradi di libertà dell’ambiente (cioè action_space),
        # che sarà proprio la mean della distribuzione di azione => l’azione “più probabile”
        # Durante l’apprendimento, voglio regolare i pesi dei layer in modo che mu(s) si sposti verso
        # le azioni che producono ritorni maggiori

        self.sigma_activation = F.softplus  # Funzione di trasformazione per garantire che la deviazione
                                            # standard sigma sia sempre positiva
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)
        # crea un tensore di zeri di lunghezza pari al numero di azioni.
	    # •	"+ init_sigma" lo riempie con 0.5
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    # Trasforma un batch di stati x in una distribuzione di azioni
    def forward(self, x):  # Input: stato s di dimensione state_space
        x = self.tanh(self.fc1_actor(x))
        x = self.tanh(self.fc2_actor(x))
        action_mean = self.fc3_actor_mean(x)

        sigma = self.sigma_activation(self.sigma)
        return Normal(action_mean, sigma)  # Restituisce un oggetto torch.distributions.Normal, che incapsula:
	                                       # •	loc= action_mean
	                                       # •	scale= sigma
	                                       # Con esso puoi campionare azioni (.sample()) e calcolare le
                                           # log-probabilities (.log_prob(...)).

# La classe prende in input:
# - policy: un’istanza di Policy, responsabile di generare la distribuzione di azione
# - critic=None: opzionale, un’istanza di Critic per l’algoritmo Actor-Critic;
#   se lasciato None, si userà solo REINFORCE
# - device='cpu': device su cui eseguire i calcoli PyTorch, può essere "cpu" oppure "cuda"
class Agent(object):
    def __init__(self, policy, max_action, critic=None, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)  # Questo garantisce che i forward e backward
                                                    # avvengano tutti sullo stesso hardware
        self.critic = critic.to(self.train_device) if critic is not None else None
        self.max_action = max_action # valore massimo dell'azione che l'agent può compiere
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        # Crea un ottimizzatore Adam per i parametri della policy
        self.optimizer_critic = (torch.optim.Adam(self.critic.parameters(), lr=1e-3)
                                 if self.critic is not None else None)  # Se esiste un critic, ne crea un
                                                                        # ottimizzatore Adam analogo
        
        # Buffer per raccogliere, passo dopo passo, all’interno di un episodio:
        self.gamma = 0.99
        self.states = []  # •	states: stati s_t
        self.next_states = []  # •	next_states: stati successivi s_{t+1}
        self.action_log_probs = []  # •	action_log_probs: i logaritmi delle probabilità delle azioni campionate
        self.rewards = []  # •	rewards: ricompense scalari r_t
        self.done = []  # •	done: flag booleani che indicano la terminazione dell’episodio

    # -------------------------------------------------------------- #
    # 1.  PICK ACTION                                                #
    # -------------------------------------------------------------- #
    # Il metodo get_action ha due compiti principali:
	# 1. Interfacciarsi con l’ambiente: riceve uno stato state (array NumPy) dall’ambiente e lo trasforma
    #    nell’azione da eseguire.
	# 2. Fornire informazioni di training: restituisce anche il log‐probability dell’azione, fondamentale
    #    per calcolare la loss del policy gradient.
    def get_action(self, state, evaluation=False):  # evaluation (bool): se True, viene usata la politica
                                                    # in modalità “deterministica” (solo la media), utile
                                                    # in fase di test o valutazione
        x = torch.from_numpy(state).float().to(self.train_device)  # Input del forward della rete di policy
        dist = self.policy(x)  # dist rappresenta la distribuzione di probabilità da cui campionare le azioni

        if evaluation:  # Non vogliamo esplorazione casuale, ma l’azione “più probabile”: la media
            action = torch.tanh(dist.mean) * self.max_action
            return action.detach().cpu().numpy(), None  # .detach() scollega il tensore dal grafo computazionale
                                                           # (non servono gradienti)

        pre_tanh_action = dist.sample()  # Estrae un’azione casuale
        tanh_action = torch.tanh(pre_tanh_action) # Comprimi le azioni tra [-1, 1]
        action = tanh_action * self.max_action # Adatta il range tra [-self.max_action, self.max_action]

        # Compute log-prob with correction
        log_prob = dist.log_prob(pre_tanh_action) # Le log probabilities devono essere calcolate dal sampling della normale
        log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6) # Deriva dalla formula log p(a) = log p(u) - \sum_i log(1-tanh^2(ui)); epsilon = 1e-6 evita log(0)
        log_prob = log_prob.sum(dim = -1, keepdim = True) # calcola la somma delle log probabilities

        # -------- PATCH: detach prima di numpy() --------
        return action.detach().cpu().numpy(), log_prob
        # Output: - l’azione pronta per l’ambiente, senza traccia di gradiente
        #         - log_prob: tensore PyTorch, verrà usato più tardi in update_policy per costruire
        #           la loss del policy gradient

    # -------------------------------------------------------------- #
    # 2.  STORE STEP                                                 #
    # -------------------------------------------------------------- #
    # Il metodo store_outcome ha il compito di accumulare, passo dopo passo, tutte le informazioni
    # necessarie per poi aggiornare la policy (e il critic) in un unico momento
    def store_outcome(self, state, next_state, log_prob, reward, done):
        # Parametri in ingresso:
	    #   • state: lo stato corrente s_t prima di eseguire l’azione.
	    #   • next_state: lo stato risultante s_{t+1} dopo aver applicato l’azione.
	    #   • log_prob: \log \pi(a_t \mid s_t), il log‐probability dell’azione campionata.
	    #   • reward: la ricompensa scalare r_t ottenuta per quella transizione.
	    #   • done: flag booleano che indica se l’episodio è terminato dopo questo passo.
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.done.append(done)

    # -------------------------------------------------------------- #
    # 3.  UPDATE                                                     #
    # -------------------------------------------------------------- #
    # raccoglie i dati accumulati durante gli step, li trasforma in batch, azzera i buffer e quindi,
    # in base all’algoritmo scelto, calcola le loss e aggiorna i parametri di policy e, se presente,
    # di critic
    def update_policy(self, model):
        log_probs = torch.stack(self.action_log_probs).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states).to(self.train_device)
        next_states = torch.stack(self.next_states).to(self.train_device)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze(-1)
        done = torch.tensor(self.done, dtype=torch.float32, device=self.train_device)
        # torch.stack(...): prende la lista di tensori (uno per passo) e la unisce lungo la prima
        # dimensione, formando un batch di forma [T, …], dove T è il numero di transizioni raccolte.

        # svuota buffer
        # Dopo aver creato i batch, le liste interne vengono resettate a vuoto in modo da raccogliere
        # i dati del prossimo episodio (o sotto-batch).
        self.states, self.next_states = [], []
        self.action_log_probs, self.rewards, self.done = [], [], []

        # ---------------- REINFORCE ---------------- #
        if model == "REINFORCE":
            disc_returns = discount_rewards(rewards, self.gamma)
            actor_loss = -(log_probs * disc_returns).sum()

            self.optimizer.zero_grad()  # azzera i gradienti precedenti
            actor_loss.backward()  # calcola i nuovi gradienti.
            self.optimizer.step()  # aggiorna i pesi della policy
            return actor_loss

        # --------------- ACTOR‑CRITIC -------------- #
        elif model == "ActorCritic":
            # --- 1) Critic update (prima) -----------
            with torch.no_grad():  # disabilita il tracciamento dei gradienti perché non vogliamo aggiornare
                                   # né la policy né il critic usando il termine futuro come nodo di back-prop
                next_act = self.policy(next_states).mean
                # qui viene chiamata critic.forward con self.critic() per calcolare il target:
                target_q = rewards + self.gamma * \
                           self.critic(next_states, next_act) * (1 - done)
                # Il fattore (1 - done) assicura che, se l’episodio è terminato, non si faccia bootstrapping oltre

            current_act = self.policy(states).mean.detach()    # grad OFF verso policy
            q_vals = self.critic(states, current_act)  # qui viene chiamata critic.forward con self.critic()
                                                       # per ottenere la stima corrente da confrontare col target

            critic_loss = F.mse_loss(q_vals, target_q)

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            # --- 2) Actor update (dopo) --------------
            advantages = (target_q - q_vals).detach()
            actor_loss = -(log_probs * advantages).sum()

            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()

            return actor_loss + critic_loss

        else:
            raise ValueError(f"Unknown algorithm '{model}'")