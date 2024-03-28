import os
import numpy as np
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datetime import datetime

from models.vpu import (
    train,
    evaluate,
    NetworkPhi,
    save_checkpoint,
)

from data.covid_data_process import (
    BertDataset,
    parse_data,
    pu_label_process_trans,
    read_ris_file,
)

from utils import (
    get_metric,
    log_metrics,
    set_seed,
)


def train_vpu_on_covid(
    data_dir,
    settings_mode,
    num_lp,
    random_state,
    batch_size,
    bertmodelpath,
    filepath,
    learning_rate,
    lam,
    mix_alpha,
    epochs,
    val_iterations,
    # hidden_size,
    # EPOCHS,
    # post_lr,
    # prior_lr,
    # cls_loss_weight,
    # post_loss_weight,
    # prior_loss_weight,
    # covid_models,
    # runs_dir,
):
    set_seed(random_state)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    Bertmodel = AutoModel.from_pretrained(bertmodelpath).to(device)
    tokenizer = AutoTokenizer.from_pretrained(bertmodelpath)
    TrainingIncludes = read_ris_file(data_dir, "1_Training_Included_20878.ris.txt")
    TrainingExcludes = read_ris_file(data_dir, "2_Training_Excluded_38635.ris.txt")
    CalibrationIncludes = read_ris_file(data_dir, "3_Calibration_Included_6005.ris.txt")
    CalibrationExcludes = read_ris_file(
        data_dir, "4_Calibration_Excluded_10118.ris.txt"
    )
    EvaluationIncludes = read_ris_file(data_dir, "5_Evaluation_Included_2310.ris.txt")
    EvaluationExcludes = read_ris_file(data_dir, "6_Evaluation_Excluded_2412.ris.txt")

    if settings_mode == 1:
        CalibrationDf = parse_data(TrainingIncludes, TrainingExcludes)
        TrainingDf = parse_data(CalibrationIncludes, CalibrationExcludes)
        EvaluationDf = parse_data(EvaluationIncludes, EvaluationExcludes)
    elif settings_mode == 2:
        EvaluationDf = parse_data(TrainingIncludes, TrainingExcludes)
        CalibrationDf = parse_data(CalibrationIncludes, CalibrationExcludes)
        TrainingDf = parse_data(EvaluationIncludes, EvaluationExcludes)
    elif settings_mode == 3:
        TrainingDf = parse_data(TrainingIncludes, TrainingExcludes)
        CalibrationDf = parse_data(CalibrationIncludes, CalibrationExcludes)
        EvaluationDf = parse_data(EvaluationIncludes, EvaluationExcludes)

    tr_df, _, _ = pu_label_process_trans(
        TrainingDf, CalibrationDf, EvaluationDf, num_lp, random_state
    )
    train_labels = tr_df.query("tr == 1")["pulabel"].values
    train_LP_index = tr_df.query("tr == 1" and "pulabel == 1").index
    val_lp_index = tr_df.query("ca == 1").index
    test_index = tr_df.query("ts == 1").index
    val_index = np.concatenate((val_lp_index, test_index), axis=0)
    val_p_labels = tr_df.query("ca == 1")["label"].values
    test_labels_val = tr_df.query("ts == 1")["pulabel"].values
    val_labels = np.concatenate((val_p_labels, test_labels_val), axis=0)
    test_labels = tr_df.query("ts == 1")["label"].values

    tr_df_concatenated = tr_df["title"] + " " + tr_df["abstract"]
    tr_inputs = tokenizer(
        tr_df_concatenated.tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    train_dataset = BertDataset(tr_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

    tr_lp_inputs = tokenizer(
        tr_df_concatenated[train_LP_index].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    train_LP_dataset = BertDataset(tr_lp_inputs, train_labels[train_LP_index])
    train_LP_loader = DataLoader(
        train_LP_dataset, batch_size=batch_size, drop_last=True
    )

    eval_inputs = tokenizer(
        tr_df_concatenated[val_index].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    eval_dataset = BertDataset(eval_inputs, val_labels)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    eval_p_inputs = tokenizer(
        tr_df_concatenated[val_lp_index].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    eval_p_dataset = BertDataset(eval_p_inputs, val_p_labels)
    eval_p_loader = DataLoader(eval_p_dataset, batch_size=batch_size, shuffle=False)

    test_inputs = tokenizer(
        tr_df_concatenated[test_index].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    test_dataset = BertDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(filepath, exist_ok=True)
    model_phi = NetworkPhi().cuda()

    opt_phi = torch.optim.Adam(
        model_phi.parameters(), lr=learning_rate, betas=(0.5, 0.99)
    )

    lowest_val_var = math.inf  # lowest variational loss on validation set
    highest_test_acc = -1  # highest test accuracy on test set
    for epoch in tqdm(range(epochs)):
        # adjust the optimizer
        if epoch % 20 == 19:
            learning_rate /= 2
            opt_phi = torch.optim.Adam(
                model_phi.parameters(), lr=learning_rate, betas=(0.5, 0.99)
            )
        # train the model \Phi
        phi_loss, var_loss, reg_loss, phi_p_mean, phi_x_mean = train(
            model_phi,
            opt_phi,
            train_LP_loader,
            train_loader,
            Bertmodel,
            val_iterations,
            mix_alpha,
            lam,
        )

        # evaluate the model \Phi
        val_var, test_acc, test_auc = evaluate(
            model_phi,
            train_loader,
            test_dataloader,
            eval_p_loader,
            eval_dataloader,
            epoch,
            phi_loss,
            var_loss,
            reg_loss,
            Bertmodel,
        )

        # assessing performance of the current model and decide whether to save it
        is_val_var_lowest = val_var < lowest_val_var
        is_test_acc_highest = test_acc > highest_test_acc
        lowest_val_var = min(lowest_val_var, val_var)
        highest_test_acc = max(highest_test_acc, test_acc)
        if is_val_var_lowest:
            test_auc_of_best_val = test_auc
            test_acc_of_best_val = test_acc
            epoch_of_best_val = epoch
        # save_checkpoint(
        #     {
        #         "epoch": epoch + 1,
        #         "state_dict": model_phi.state_dict(),
        #         "optimizer": opt_phi.state_dict(),
        #     },
        #     is_val_var_lowest,
        #     is_test_acc_highest,
        #     filepath,
        # )

    # inform users model in which epoch is finally picked
    print(
        "Early stopping at {:}th epoch, test AUC : {:.4f}, test acc: {:.4f}".format(
            epoch_of_best_val, test_auc_of_best_val, test_acc_of_best_val
        )
    )


if __name__ == "__main__":
    train_vpu_on_covid(
        data_dir=r"/root/autodl-tmp/PU_all_in_one/data/Cochrane_Covid-19",
        settings_mode=3,
        num_lp=50,
        random_state=42,
        batch_size=32,
        bertmodelpath=r"/root/autodl-tmp/PU_all_in_one/pretrained/allenai/scibert_scivocab_uncased",
        filepath=r"/root/autodl-tmp/PU_all_in_one/saved_models/covid_task/VPU/vpu_model_ckpt.pth",
        learning_rate=3e-5,
        lam=0.1,
        mix_alpha=0.1,
        epochs=10,
        val_iterations=20,
    )
