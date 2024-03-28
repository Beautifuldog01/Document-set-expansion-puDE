import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from nltk.tokenize import word_tokenize
from collections import Counter
import itertools


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_reduction(preds, labels):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    recall = recall_score(labels, preds)
    N = len(labels)
    work_reduction = (tn + fn) / (N * (1 - recall + 1e-8))

    return work_reduction


def plot_confusion_matrix(
        cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def cal_metric(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    return f1, recall


def get_threshold(logits, labels, n_min, p_max, step=0.001, patience=20):
    thresholds = np.arange(n_min, p_max, step)
    preds = torch.gt(
        logits.unsqueeze(-1), torch.from_numpy(thresholds).float().to(logits.device)
    ).float()

    f1_scores = [f1_score(labels, pred.squeeze()) for pred in preds.T]

    # Early stopping
    no_improve = 0
    max_f1 = 0.0
    best_threshold = n_min
    for i, f1 in enumerate(f1_scores):
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = thresholds[i]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_threshold


def get_threshold_99(probs, labels):
    positive_probs = probs[labels == 1]
    threshold99 = np.quantile(positive_probs, 0.05)
    return threshold99


def find_last_label_position(prob, labels):
    sorted_indices = np.argsort(-prob)
    prob_labels_1 = prob[labels == 1]
    last_prob_label_1 = np.min(prob_labels_1)
    position = np.where(prob[sorted_indices] == last_prob_label_1)[0][0] + 1
    return position


def calculate_precision_recall_at_k(labels, prob, k=10):
    top_k_percent = int(len(labels) * (k / 100.0))
    combined = list(zip(labels, prob))
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
    top_k = combined_sorted[:top_k_percent]
    precision_at_k = sum([1 for label, _ in top_k if label == 1]) / top_k_percent
    true_positives = sum([1 for label, _ in top_k if label == 1])
    all_positives = sum(labels)
    recall_at_k = true_positives / all_positives if all_positives > 0 else 0

    return precision_at_k, recall_at_k


def calculate_wss(scores, labels, r=95):
    """
    Calculate WSS@r% for given prediction scores and true labels.

    Parameters:
    - scores: A list of prediction scores or probabilities for being the positive class.
    - labels: A list of true labels, where 1 represents the positive class and 0 represents the negative class.
    - r: The recall level at which WSS is calculated. Default is 95 for WSS@95%.

    Returns:
    - WSS@r% value.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    idx = np.argsort(scores)[::-1]
    sorted_labels = labels[idx]
    cumsum_labels = np.cumsum(sorted_labels)
    total_positives = np.sum(labels)
    recall_at_i = cumsum_labels / total_positives
    target_recall = r / 100.0
    documents_to_review = np.argmax(recall_at_i >= target_recall) + 1
    total_documents = len(scores)
    wss_r = (1 - documents_to_review / total_documents) * 100

    return wss_r


def get_metric(labels, prob):
    labels = (
        labels.cpu() if isinstance(labels, torch.Tensor) and labels.is_cuda else labels
    )
    prob = prob.cpu() if isinstance(prob, torch.Tensor) and prob.is_cuda else prob

    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    prob = prob.numpy() if isinstance(prob, torch.Tensor) else prob
    auc = roc_auc_score(labels, prob)
    r_10, r_20, r_30, r_40, r_50, r_95 = get_rec(labels, prob)
    WSS95 = calculate_wss(prob, labels, r=95)
    WSS100 = calculate_wss(prob, labels, r=100)
    p_mean = np.mean(prob[labels == 1])
    n_mean = np.mean(prob[labels == 0])
    p_LastRel = find_last_label_position(prob, labels)

    prob = torch.tensor(prob)
    threshold = get_threshold(prob, labels, n_mean, p_mean, step=0.001)
    threshold99 = get_threshold_99(prob, labels)
    preds = torch.gt(prob, threshold).float()
    preds99 = torch.gt(prob, threshold99).float()
    f1, rec = cal_metric(labels, preds)
    # if mode == "test":
    #     # cm = confusion_matrix(labels, preds)
    #     # plot_confusion_matrix(cm, classes=['False', 'True'],
    #     #                 title='Confusion matrix, without normalization')
    #     # plt.show()
    #     pass
    acc = accuracy_score(labels, preds)
    prec10, recall10 = calculate_precision_recall_at_k(labels, prob, k=10)
    prec20, recall20 = calculate_precision_recall_at_k(labels, prob, k=20)

    f1_99, rec_99 = cal_metric(labels, preds99)
    acc_99 = accuracy_score(labels, preds99)

    reduce_work = get_reduction(preds99, labels)

    return (
        threshold,
        threshold99,
        auc,
        f1,
        acc,
        rec,
        f1_99,
        acc_99,
        rec_99,
        r_10,
        r_20,
        r_30,
        r_40,
        r_50,
        r_95,
        reduce_work,
        p_mean,
        n_mean,
        WSS95,
        WSS100,
        p_LastRel,
        prec10,
        prec20,
        recall10,
        recall20,
    )


def get_rec(labels, logits):
    pos_logits = logits[labels == 1]
    sorted_logits = sorted(logits, reverse=True)
    s_p = sorted(pos_logits)

    TOP10 = sorted_logits[int(len(sorted_logits) * 0.1) - 1]
    TOP20 = sorted_logits[int(len(sorted_logits) * 0.2) - 1]
    TOP30 = sorted_logits[int(len(sorted_logits) * 0.3) - 1]
    TOP40 = sorted_logits[int(len(sorted_logits) * 0.4) - 1]
    TOP50 = sorted_logits[int(len(sorted_logits) * 0.5) - 1]
    TOP95 = sorted_logits[int(len(sorted_logits) * 0.95) - 1]
    r_10 = sum(1 for i in s_p if i >= TOP10) / len(s_p)
    r_20 = sum(1 for i in s_p if i >= TOP20) / len(s_p)
    r_30 = sum(1 for i in s_p if i >= TOP30) / len(s_p)
    r_40 = sum(1 for i in s_p if i >= TOP40) / len(s_p)
    r_50 = sum(1 for i in s_p if i >= TOP50) / len(s_p)
    r_95 = sum(1 for i in s_p if i >= TOP95) / len(s_p)
    return r_10, r_20, r_30, r_40, r_50, r_95


def calculate_accuracy(y_true, y_pred):
    # (TP + TN) / (TP + TN + FP + FN)
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN = np.sum(np.logical_and(y_true == -1, y_pred == -1))
    FP = np.sum(np.logical_and(y_true == -1, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == -1))
    return (TP + TN) / (TP + TN + FP + FN)


def getFeatures(data, word_to_index, max_length):
    all_features = []

    for index in range(len(data)):
        title_tokens = data.title[index].split()
        abstract_tokens = data.abstract[index].split()

        title_indices = [word_to_index.get(word.lower(), 0) for word in title_tokens]
        abstract_indices = [
            word_to_index.get(word.lower(), 0) for word in abstract_tokens
        ]

        title_indices += [0] * (max_length - len(title_indices))
        title_indices = title_indices[:max_length]
        abstract_indices += [0] * (max_length - len(abstract_indices))
        abstract_indices = abstract_indices[:max_length]

        all_features.append((title_indices, abstract_indices))

    return all_features


def log_metrics(writer, phase, metrics, epoch):
    """
    Log metrics using TensorBoard.
    """
    (
        threshold,
        threshold99,
        auc,
        f1,
        acc,
        rec,
        f1_99,
        acc_99,
        rec_99,
        r_10,
        r_20,
        r_30,
        r_40,
        r_50,
        r_95,
        reduce_work,
        p_mean,
        n_mean,
        WSS95,
        WSS100,
        LastRel,
        prec10,
        prec20,
        recall10,
        recall20,
    ) = metrics

    writer.add_scalar(f"{phase}/AUC", auc, epoch)
    writer.add_scalar(f"{phase}/F1", f1, epoch)
    writer.add_scalar(f"{phase}/Accuracy", acc, epoch)
    writer.add_scalar(f"{phase}/Recall", rec, epoch)
    # writer.add_scalar(f"{phase}/F1_99", f1_99, epoch)
    # writer.add_scalar(f"{phase}/Accuracy_99", acc_99, epoch)
    # writer.add_scalar(f"{phase}/Recall_99", rec_99, epoch)
    # writer.add_scalar(f"{phase}/R_10", r_10, epoch)
    # writer.add_scalar(f"{phase}/R_20", r_20, epoch)
    # writer.add_scalar(f"{phase}/R_30", r_30, epoch)
    # writer.add_scalar(f"{phase}/R_40", r_40, epoch)
    # writer.add_scalar(f"{phase}/R_50", r_50, epoch)
    # writer.add_scalar(f"{phase}/R_95", r_95, epoch)
    # writer.add_scalar(f"{phase}/Reduce_Work", reduce_work, epoch)
    # writer.add_scalar(f"{phase}/Positive_Mean", p_mean, epoch)
    # writer.add_scalar(f"{phase}/Negative_Mean", n_mean, epoch)
    writer.add_scalar(f"{phase}/WSS95", WSS95, epoch)
    writer.add_scalar(f"{phase}/WSS100", WSS100, epoch)
    writer.add_scalar(f"{phase}/LastRel", LastRel, epoch)
    writer.add_scalar(f"{phase}/Precision_@10", prec10, epoch)
    writer.add_scalar(f"{phase}/Precision_@20", prec20, epoch)
    writer.add_scalar(f"{phase}/Recall_@10", recall10, epoch)
    writer.add_scalar(f"{phase}/Recall_@20", recall20, epoch)


def build_vocab(texts, min_freq=2):
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    counter = Counter(itertools.chain.from_iterable(tokenized_texts))

    vocab = {
        word: i + 2
        for i, (word, freq) in enumerate(counter.items())
        if freq >= min_freq
    }

    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    return vocab


def print_metrics(metrics, info_tuple):
    for i, metric in enumerate(metrics):
        print(f"{metric}: {info_tuple[i]}")
    print("\n")
