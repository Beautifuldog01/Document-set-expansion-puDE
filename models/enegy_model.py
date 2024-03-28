import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd


class EnergyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnergyNet, self).__init__()
        self.first = nn.Linear(input_dim, 512)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 128)
        self.last = nn.Linear(128, output_dim)
        self.b1 = nn.BatchNorm1d(512)
        self.b2 = nn.BatchNorm1d(256)
        self.b3 = nn.BatchNorm1d(128)
        self.b4 = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU()
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.b1(self.first(x))
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        x = self.b2(self.linear1(x))
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        x = self.b3(self.linear2(x))
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        x = self.b4(self.linear3(x))
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        x = self.last(x)
        # return torch.sigmoid(x)
        return x


class NonNegativePULoss(nn.Module):
    def __init__(self, prior, positive_class=1, loss=None, gamma=1, beta=0, nnpu=True):
        super(NonNegativePULoss, self).__init__()
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss = loss or (lambda x: torch.sigmoid(-x))
        self.nnPU = nnpu
        self.positive = positive_class
        self.unlabeled = 1 - positive_class

    def forward(self, x, t):
        t = t[:, None]
        positive, unlabeled = (t == self.positive).float(), (
            t == self.unlabeled
        ).float()
        n_positive, n_unlabeled = max(1.0, positive.sum().item()), max(
            1.0, unlabeled.sum().item()
        )

        y_positive = self.loss(x)  # per sample positive risk
        y_unlabeled = self.loss(-x)  # per sample negative risk

        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled
        )

        if self.nnPU:
            if negative_risk.item() < -self.beta:
                objective = (
                    positive_risk - self.beta + self.gamma * negative_risk
                ).detach() - self.gamma * negative_risk
            else:
                objective = positive_risk + negative_risk
        else:
            objective = positive_risk + negative_risk

        return objective


def infer_prob(post_model, prior_model, x):
    post_prob = post_model(x).squeeze()
    prior_prob = prior_model(x).squeeze()
    # prob = torch.exp(post_prob - prior_prob)
    prob = post_prob - prior_prob
    # prob = torch.where(torch.isnan(prob), torch.full_like(prob, 0), prob)
    prob = torch.clamp(prob, -100.0, 100.0)
    return prob


def sample_langevin(
    x, model, stepsize, n_steps, noise_scale=None, intermediate_samples=False
):
    """Draw samples using Langevin dynamics
    x: torch.Tensor, initial points
    model: An energy-based model
    stepsize: float
    n_steps: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    """
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)

    l_samples = []
    l_dynamics = []
    x.requires_grad = True
    for _ in range(n_steps):
        l_samples.append(x.detach().to("cpu"))
        noise = torch.randn_like(x) * noise_scale
        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        dynamics = stepsize * grad + noise
        x = x + dynamics
        l_samples.append(x.detach().to("cpu"))
        l_dynamics.append(dynamics.detach().to("cpu"))

    if intermediate_samples:
        return l_samples, l_dynamics
    else:
        return l_samples[-1]


def random_replace_input_ids(input_ids, replacement_prob=0.1, vocab_size=31090):
    random_probs = torch.rand(input_ids.shape)
    random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape)

    mask = random_probs < replacement_prob
    input_ids_replaced = torch.where(mask, random_tokens, input_ids)

    return input_ids_replaced
