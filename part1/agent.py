import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    """Compute Monte-Carlo discounted returns G_t."""
    discounted_r = torch.zeros_like(r)
    running_add = 0.0
    for t in reversed(range(r.size(0))):
        running_add = r[t] + gamma * running_add
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden=64):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.tanh = torch.nn.Tanh()

        # Actor network: state -> mean of a Gaussian policy over continuous actions
        self.fc1_actor = torch.nn.Linear(state_space, hidden)
        self.fc2_actor = torch.nn.Linear(hidden, hidden)
        self.fc3_actor_mean = torch.nn.Linear(hidden, action_space)

        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(action_space) + init_sigma)

        # Critic network: state -> scalar V(s), used only for Actor-Critic
        self.fc1_critic = torch.nn.Linear(state_space, hidden)
        self.fc2_critic = torch.nn.Linear(hidden, hidden)
        self.fc3_critic_value = torch.nn.Linear(hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """Return both the stochastic actor distribution and V(s)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Actor
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma) + 1e-5
        normal_dist = Normal(action_mean, sigma)

        # Critic
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic).squeeze(-1)

        if squeeze_output:
            
            state_value = state_value.squeeze(0)

        return normal_dist, state_value


class Agent(object):
    def __init__(
        self,
        policy,
        algorithm="reinforce",
        baseline=None,
        device="cpu",
        gamma=0.99,
        lr=1e-3,
        value_coef=0.5,
        entropy_coef=0.0,
    ):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.algorithm = algorithm
        self.baseline = baseline
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        if len(self.rewards) == 0:
            return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0}

        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.tensor(self.done, dtype=torch.float32, device=self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        if self.algorithm in ["reinforce", "reinforce_baseline"]:
            # Monte-Carlo return G_t = r_t + gamma r_{t+1} + ...
            returns = discount_rewards(rewards, self.gamma)

            
            baseline_value = 0.0 if self.baseline is None else float(self.baseline)
            advantages = returns - baseline_value

        
            policy_loss = -(action_log_probs * advantages.detach()).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            return {
                "loss": float(policy_loss.detach().cpu()),
                "actor_loss": float(policy_loss.detach().cpu()),
                "critic_loss": 0.0,
            }

        elif self.algorithm == "actor_critic":
            # Critic estimate V(s_t)
            _, values = self.policy(states)

            
            with torch.no_grad():
                _, next_values = self.policy(next_states)
                targets = rewards + self.gamma * next_values * (1.0 - done)
                advantages = targets - values

            actor_loss = -(action_log_probs * advantages.detach()).mean()
            critic_loss = F.mse_loss(values, targets)

            
            normal_dist, _ = self.policy(states)
            entropy = normal_dist.entropy().sum(dim=-1).mean()

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            return {
                "loss": float(loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
            }

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def get_action(self, state, evaluation=False):
        """Return action as numpy array and log-probability tensor."""
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist, _ = self.policy(x)

        if evaluation:
            action = normal_dist.mean.squeeze(0)
            return action.detach().cpu().numpy(), None

        action = normal_dist.sample().squeeze(0)
        action_log_prob = normal_dist.log_prob(action).sum()
        return action.detach().cpu().numpy(), action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.done.append(done)
