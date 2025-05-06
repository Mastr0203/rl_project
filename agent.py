"""
agent.py
--------

* Rimane indipendente da Gym ↔ Gymnasium: gli unici oggetti che riceve
  sono `state` (np.ndarray), `reward` (float) e un flag `done`.
* Non è quindi necessario alcun refactor strutturale per il passaggio
  a Gymnasium + MuJoCo; aggiorniamo solo:
    - import / commenti
    - segnaposto chiari per i TASK 2 e 3
"""

from __future__ import annotations  # Permette di usare “tipi” come stringhe (es. List["Tensor"])
                                    # anche prima che siano completamente definiti

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


# --------------------------------------------------------------------- #
# Utils                                                                 #
# --------------------------------------------------------------------- #

def discount_rewards(r: Tensor, gamma: float) -> Tensor:
    """Return discounted cumulative rewards (G_t)."""
    discounted_r = torch.zeros_like(r)  # Alloca un tensor vuoto e inizializza la somma cumulativa
    running_add = 0.0
    for t in reversed(range(r.size(0))):  # Cicla all’indietro nel tempo per accumulare reward scontati
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r  # Ritorna il tensore dei return G_t per ogni timestep

# Il return G_t viene usato per pesare il gradiente log-probabilità
# → Se G_t è alto → rinforza l’azione scelta
# → Se G_t è basso → scoraggia quella scelta


# --------------------------------------------------------------------- #
# Policy network (actor + critic skeleton)                              #
# --------------------------------------------------------------------- #

class Policy(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # --- Actor ----------------------------------------------------- #
        self.fc1_actor = torch.nn.Linear(state_dim, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_dim)
        self.sigma = torch.nn.Parameter(torch.full((action_dim,), 0.5))
        self.sigma_activation = F.softplus

        # --- Critic ----------------------------------------------------- #
        self.fc1_critic = torch.nn.Linear(state_dim, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_value = torch.nn.Linear(self.hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> tuple[Normal, Tensor]:
        # Actor
        a = self.tanh(self.fc1_actor(x))
        a = self.tanh(self.fc2_actor(a))
        mean = self.fc3_actor_mean(a)
        sigma = self.sigma_activation(self.sigma)
        dist = Normal(mean, sigma)

        # Critic
        v = self.tanh(self.fc1_critic(x))
        v = self.tanh(self.fc2_critic(v))
        value = self.fc3_value(v).squeeze(-1)

        return dist, value


class Agent:
    def __init__(self, policy: Policy, device: str = "cpu"):
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.device = device
        self.gamma = 0.99

        self.states: List[Tensor] = []
        self.action_log_probs: List[Tensor] = []
        self.rewards: List[Tensor] = []
        self.dones: List[bool] = []

    def get_action(self, state: np.ndarray, evaluation: bool = False):
        x = torch.from_numpy(state).float().to(self.device)
        dist, _ = self.policy(x)

        if evaluation:
            return dist.mean.cpu().numpy(), None
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            return action.cpu().numpy(), log_prob

    def store_outcome(self, state: np.ndarray, _: np.ndarray, log_prob: Tensor, reward: float, done: bool):
        self.states.append(torch.from_numpy(state).float())
        self.action_log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.dones.append(done)

    def update_policy(self, method: str = "REINFORCE") -> float:
        log_probs = torch.stack(self.action_log_probs).to(self.device)
        rewards = torch.stack(self.rewards).to(self.device).flatten()
        states = torch.stack(self.states).to(self.device)

        self.states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

        if method == "REINFORCE":
            returns = discount_rewards(rewards, self.gamma)
            loss = -torch.sum(log_probs * returns)

        elif method == "ActorCritic":
            _, values = self.policy(states)
            returns = discount_rewards(rewards, self.gamma)
            advantages = returns - values.detach()

            loss_actor = -torch.sum(log_probs * advantages)
            loss_critic = F.mse_loss(values, returns)
            loss = loss_actor + loss_critic

        else:
            raise ValueError(f"Unknown method: {method}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()