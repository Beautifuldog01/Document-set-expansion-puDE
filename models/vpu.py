import math
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import errno
import shutil
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from utils import (
    get_metric,
    set_seed,
    print_metrics,
)


def save_checkpoint(state, is_lowest_on_val, is_highest_on_test, filepath):
    # dir_path = os.path.dirname(filepath)
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    # torch.save(state, filepath)
    # if is_lowest_on_val:
    #     shutil.copyfile(filepath, "model_lowest_on_val.pth.tar")
    # if is_highest_on_test:
    #     shutil.copyfile(filepath, "model_highest_on_test.pth.tar")
    pass


class AverageMeter(object):
    """
    Computes and stores the average and current value

    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(
    model_phi, opt_phi, p_loader, x_loader, bertmodel, val_iterations, mix_alpha, lam
):
    """
    One epoch of the training of VPU.

    :param config: arguments.
    :param model_phi: current model \Phi.
    :param opt_phi: optimizer of \Phi.
    :param p_loader: loader for the labeled positive training data.
    :param x_loader: loader for training data (including positive and unlabeled)
    """

    # setup some utilities for analyzing performance
    phi_p_avg = AverageMeter()
    phi_x_avg = AverageMeter()
    phi_loss_avg = AverageMeter()
    var_loss_avg = AverageMeter()
    reg_avg = AverageMeter()
    x_iter = cycle(x_loader)
    p_iter = cycle(p_loader)
    # set the model to train mode
    model_phi.train()
    for batch_idx in range(val_iterations):
        data_x, _ = next(x_iter)
        data_p, _ = next(p_iter)

        if torch.cuda.is_available():
            with torch.no_grad():
                data_p = torch.mean(
                    bertmodel(
                        data_p["input_ids"].cuda(),
                        data_p["attention_mask"].cuda(),
                        data_p["token_type_ids"].cuda(),
                    ).last_hidden_state,
                    dim=1,
                )
                data_x = torch.mean(
                    bertmodel(
                        data_x["input_ids"].cuda(),
                        data_x["attention_mask"].cuda(),
                        data_x["token_type_ids"].cuda(),
                    ).last_hidden_state,
                    dim=1,
                ).cuda()

        # calculate the variational loss
        data_all = torch.cat((data_p, data_x))
        output_phi_all = model_phi(data_all)
        log_phi_all = output_phi_all[:, 1]
        idx_p = slice(0, len(data_p))
        idx_x = slice(len(data_p), len(data_all))
        log_phi_x = log_phi_all[idx_x]
        log_phi_p = log_phi_all[idx_p]
        output_phi_x = output_phi_all[idx_x]
        var_loss = (
            torch.logsumexp(log_phi_x, dim=0)
            - math.log(len(log_phi_x))
            - 1 * torch.mean(log_phi_p)
        )

        # perform Mixup and calculate the regularization
        target_x = output_phi_x[:, 1].exp()
        target_p = torch.ones(len(data_p), dtype=torch.float32)
        target_p = target_p.cuda() if torch.cuda.is_available() else target_p
        rand_perm = torch.randperm(data_p.size(0))
        data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
        m = torch.distributions.beta.Beta(mix_alpha, mix_alpha)
        lam = m.sample()
        data = lam * data_x + (1 - lam) * data_p_perm
        target = lam * target_x + (1 - lam) * target_p_perm
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        out_log_phi_all = model_phi(data)
        reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()

        # calculate gradients and update the network
        phi_loss = var_loss + lam * reg_mix_log
        opt_phi.zero_grad()
        phi_loss.backward()
        opt_phi.step()

        # update the utilities for analysis of the model
        reg_avg.update(reg_mix_log.item())
        phi_loss_avg.update(phi_loss.item())
        var_loss_avg.update(var_loss.item())
        phi_p, phi_x = log_phi_p.exp(), log_phi_x.exp()
        phi_p_avg.update(phi_p.mean().item(), len(phi_p))
        phi_x_avg.update(phi_x.mean().item(), len(phi_x))

    return phi_loss_avg.avg, var_loss_avg.avg, reg_avg.avg, phi_p_avg.avg, phi_x_avg.avg


def evaluate(
    model_phi,
    x_loader,
    test_loader,
    val_p_loader,
    val_x_loader,
    epoch,
    phi_loss,
    var_loss,
    reg_loss,
    bertmodel,
):
    """
    evaluate the performance on test set, and calculate the variational loss on validation set.

    :param model_phi: current model \Phi
    :param x_loader: loader for the whole training set (positive and unlabeled).
    :param test_loader: loader for the test set (fully labeled).
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    :param epoch: current epoch.
    :param phi_loss: VPU loss of the current epoch, which equals to var_loss + reg_loss.
    :param var_loss: variational loss of the training set.
    :param reg_loss: regularization loss of the training set.
    """

    # set the model to evaluation mode
    model_phi.eval()
    bertmodel.eval()
    # calculate variational loss of the validation set consisting of PU data
    val_var = cal_val_var(model_phi, val_p_loader, val_x_loader, bertmodel)

    # max_phi is needed for normalization
    log_max_phi = -math.inf
    for idx, (data, _) in enumerate(x_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                data = torch.mean(
                    bertmodel(
                        data["input_ids"].cuda(),
                        data["attention_mask"].cuda(),
                        data["token_type_ids"].cuda(),
                    ).last_hidden_state,
                    dim=1,
                ).cuda()
        log_max_phi = max(log_max_phi, model_phi(data)[:, 1].max())

    # feed test set to the model and calculate accuracy and AUC
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                data = torch.mean(
                    bertmodel(
                        data["input_ids"].cuda(),
                        data["attention_mask"].cuda(),
                        data["token_type_ids"].cuda(),
                    ).last_hidden_state,
                    dim=1,
                ).cuda()
            log_phi = model_phi(data)[:, 1]
            log_phi -= log_max_phi
            if idx == 0:
                log_phi_all = log_phi
                target_all = target
            else:
                log_phi_all = torch.cat((log_phi_all, log_phi))
                target_all = torch.cat((target_all, target))
    pred_all = np.array((log_phi_all > math.log(0.5)).cpu().detach())
    log_phi_all = np.array(log_phi_all.cpu().detach())
    target_all = np.array(target_all.cpu().detach())
    test_acc = accuracy_score(target_all, pred_all)
    test_auc = roc_auc_score(target_all, log_phi_all)
    test_f1 = f1_score(target_all, pred_all)
    info_tuple = get_metric(target_all, log_phi_all)
    print(
        "Train Epoch: {}\t phi_loss: {:.4f}   var_loss: {:.4f}   reg_loss: {:.4f}   Test accuracy: {:.4f}   Test F1: {:.4f}  Val var loss: {:.4f}".format(
            epoch, phi_loss, var_loss, reg_loss, test_acc, test_f1, val_var
        )
    )
    metrics = [
        "threshold",
        "threshold99",
        "auc",
        "f1",
        "acc",
        "rec",
        "f1_99",
        "acc_99",
        "rec_99",
        "r_10",
        "r_20",
        "r_30",
        "r_40",
        "r_50",
        "r_95",
        "reduce_work",
        "p_mean",
        "n_mean",
        "WSS95",
        "WSS100",
        "p_LastRel",
        "prec10",
        "prec20",
        "recall10",
        "recall20",
    ]
    print_metrics(metrics, info_tuple)

    return val_var, test_acc, test_auc


def cal_val_var(model_phi, val_p_loader, val_x_loader, bertmodel):
    """
    Calculate variational loss on the validation set, which consists of only positive and unlabeled data.

    :param model_phi: current \Phi model.
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    """

    # set the model to evaluation mode
    model_phi.eval()

    # feed the validation set to the model and calculate variational loss
    with torch.no_grad():
        for idx, (data_x, _) in enumerate(val_x_loader):
            if torch.cuda.is_available():
                data_x = torch.mean(
                    bertmodel(
                        data_x["input_ids"].cuda(),
                        data_x["attention_mask"].cuda(),
                        data_x["token_type_ids"].cuda(),
                    ).last_hidden_state,
                    dim=1,
                ).cuda()
            output_phi_x_curr = model_phi(data_x)
            if idx == 0:
                output_phi_x = output_phi_x_curr
            else:
                output_phi_x = torch.cat((output_phi_x, output_phi_x_curr))
        for idx, (data_p, _) in enumerate(val_p_loader):
            if torch.cuda.is_available():
                data_p = torch.mean(
                    bertmodel(
                        data_p["input_ids"].cuda(),
                        data_p["attention_mask"].cuda(),
                        data_p["token_type_ids"].cuda(),
                    ).last_hidden_state,
                    dim=1,
                ).cuda()
            output_phi_p_curr = model_phi(data_p)
            if idx == 0:
                output_phi_p = output_phi_p_curr
            else:
                output_phi_p = torch.cat((output_phi_p, output_phi_p_curr))
        log_phi_p = output_phi_p[:, 1]
        log_phi_x = output_phi_x[:, 1]
        var_loss = (
            torch.logsumexp(log_phi_x, dim=0)
            - math.log(len(log_phi_x))
            - torch.mean(log_phi_p)
        )
        return var_loss.item()


class NetworkPhi(nn.Module):
    def __init__(self):
        super(NetworkPhi, self).__init__()

        self.fc1 = nn.Linear(768, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.log_softmax(x)

        return x
