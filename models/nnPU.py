import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PUModel(nn.Module):
    """
    Basic Multi-layer perceptron as described in "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
    """

    def __init__(self, input_dim, hidden_size):
        super(PUModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x


class PUModelWithSciBERT(nn.Module):
    def __init__(self, model):
        super(PUModelWithSciBERT, self).__init__()
        self.sci_bert = model
        embedding_size = self.sci_bert.config.hidden_size

        self.fc1 = nn.Linear(embedding_size, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output = self.sci_bert(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state
        pooled_output = torch.mean(sequence_output, dim=1)
        x = self.fc1(pooled_output)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x
