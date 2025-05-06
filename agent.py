import numpy as np
import torch
import torch.nn.functional as F               # <‑‑ import usato solo nel critic loss
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


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

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.tanh(self.fc1_critic(x))
        x = self.tanh(self.fc2_critic(x))
        return self.fc3_critic_mean(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


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

        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.tanh(self.fc1_actor(x))
        x = self.tanh(self.fc2_actor(x))
        action_mean = self.fc3_actor_mean(x)

        sigma = self.sigma_activation(self.sigma)
        return Normal(action_mean, sigma)


class Agent(object):
    def __init__(self, policy, critic=None, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.critic = critic.to(self.train_device) if critic is not None else None

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.optimizer_critic = (torch.optim.Adam(self.critic.parameters(), lr=1e-3)
                                 if self.critic is not None else None)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    # -------------------------------------------------------------- #
    # 1.  PICK ACTION                                                #
    # -------------------------------------------------------------- #
    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        dist = self.policy(x)

        if evaluation:
            return dist.mean.detach().cpu().numpy(), None

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        # -------- PATCH: detach prima di numpy() --------
        return action.detach().cpu().numpy(), log_prob

    # -------------------------------------------------------------- #
    # 2.  STORE STEP                                                 #
    # -------------------------------------------------------------- #
    def store_outcome(self, state, next_state, log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.done.append(done)

    # -------------------------------------------------------------- #
    # 3.  UPDATE                                                     #
    # -------------------------------------------------------------- #
    def update_policy(self, model):
        log_probs = torch.stack(self.action_log_probs).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states).to(self.train_device)
        next_states = torch.stack(self.next_states).to(self.train_device)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze(-1)
        done = torch.tensor(self.done, dtype=torch.float32, device=self.train_device)

        # svuota buffer
        self.states, self.next_states = [], []
        self.action_log_probs, self.rewards, self.done = [], [], []

        # ---------------- REINFORCE ---------------- #
        if model == "REINFORCE":
            disc_returns = discount_rewards(rewards, self.gamma)
            actor_loss = -(log_probs * disc_returns).sum()

            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()
            return actor_loss

        # --------------- ACTOR‑CRITIC -------------- #
        elif model == "ActorCritic":
            # --- 1) Critic update (prima) -----------
            with torch.no_grad():
                next_act = self.policy(next_states).mean       # grad OFF
                target_q = rewards + self.gamma * \
                           self.critic(next_states, next_act) * (1 - done)

            current_act = self.policy(states).mean.detach()    # grad OFF verso policy
            q_vals = self.critic(states, current_act)

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