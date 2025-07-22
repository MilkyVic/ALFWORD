import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from alfworld.agents.agent.base_agent import BaseAgent
from alfworld.agents.modules.generic import to_pt, to_np, pad_sequences, max_len
import torch.nn.functional as F

class PPOBuffer:
    def __init__(self):
        self.obs = []
        self.tasks = []
        self.action_candidates = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.prev_dynamics = []

    def store(self, obs, task, action_candidates, action, logprob, reward, done, value, prev_dyn):
        self.obs.append(obs)
        self.tasks.append(task)
        self.action_candidates.append(action_candidates)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.prev_dynamics.append(prev_dyn)

    def clear(self):
        self.__init__()

class TextPPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        assert self.training_method == "ppo"
        self.gamma = 0.99
        self.lam = 0.95
        self.ppo_epochs = 4
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.buffer = PPOBuffer()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config['general']['training']['optimizer']['learning_rate'])

    def select_action(self, obs, task, action_candidates, previous_dynamics=None):
        self.online_net.eval()
        with torch.no_grad():
            h_obs, obs_mask = self.encode([obs], use_model="online")
            h_td, td_mask = self.encode([task], use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring([action_candidates], h_obs, obs_mask, h_td, td_mask, previous_dynamics, use_model="online")
            action_probs = torch.softmax(action_scores, dim=1)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample().item()
            device = torch.device("cuda" if self.use_cuda else "cpu")
            logprob = dist.log_prob(torch.tensor(action_idx, device=device)).item()
            value = (action_scores[0] * action_probs[0]).sum().item()  # expected value as proxy
        self.online_net.train()
        return action_candidates[action_idx], action_idx, logprob, value, current_dynamics

    def store_transition(self, obs, task, action_candidates, action_idx, logprob, reward, done, value, prev_dyn):
        self.buffer.store(obs, task, action_candidates, action_idx, logprob, reward, done, value, prev_dyn)

    def finish_path(self, last_value=0):
        rewards = np.array(self.buffer.rewards + [last_value])
        values = np.array(self.buffer.values + [last_value])
        dones = np.array(self.buffer.dones + [0])
        gae = 0
        advs = []
        for t in reversed(range(len(self.buffer.rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)
        returns = advs + values[:-1]
        self.advantages = np.array(advs, dtype=np.float32)
        self.returns = np.array(returns, dtype=np.float32)

    def update(self):
        obs = self.buffer.obs
        tasks = self.buffer.tasks
        action_candidates = self.buffer.action_candidates
        actions = np.array(self.buffer.actions)
        device = torch.device("cuda" if self.use_cuda else "cpu")
        # Đảm bảo detach hoàn toàn khỏi graph
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32, device=device).detach()
        returns = torch.tensor(np.array(self.returns), dtype=torch.float32, device=device).detach()
        advantages = torch.tensor(np.array(self.advantages), dtype=torch.float32, device=device).detach()
        prev_dynamics = self.buffer.prev_dynamics

        # Batch encode
        h_obs, obs_mask = self.encode(obs, use_model="online")
        h_td, td_mask = self.encode(tasks, use_model="online")
        # action_candidates: list of list of str
        # Pad action_candidates to same length
        max_cand = max([len(c) for c in action_candidates])
        padded_candidates = [c + ["<pad>"] * (max_cand - len(c)) for c in action_candidates]
        # Get candidate representations
        cand_repr, cand_mask = self.get_action_candidate_representations(padded_candidates, use_model="online")
        # Action indices tensor
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        # PPO update
        for _ in range(self.ppo_epochs):
            h_obs, obs_mask = self.encode(obs, use_model="online")
            h_td, td_mask = self.encode(tasks, use_model="online")
            action_scores, _, _ = self.action_scoring(padded_candidates, h_obs, obs_mask, h_td, td_mask, None, use_model="online")
            action_probs = torch.softmax(action_scores, dim=1)
            dist = torch.distributions.Categorical(action_probs)
            new_logprobs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            values = (action_scores * action_probs).sum(dim=1)
            # PPO loss
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
            self.optimizer.step()
        self.buffer.clear() 