from data.covid_data_process import (
    read_ris_file,
    parse_data,
    pu_label_process_trans,
    BiDataset,
    ProportionalSampler,
)

from utils import (
    set_seed,
    build_vocab,
    getFeatures,
    get_metric,
    log_metrics,
)

from torch.utils.tensorboard import SummaryWriter
from models.nnPU_cnn import TextClassifier, NonNegativePULoss

import os
import torch
import argparse
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime

set_seed(42)


def train_nnpu_on_covid(
    data_dir,
    settings_mode,
    num_lp,
    random_state,
    embedding_dim,
    max_length,
    batch_size,
    prior,
    learning_rate,
    num_epochs,
    covid_models,
):

    os.makedirs(covid_models, exist_ok=True)
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
    else:
        TrainingDf, CalibrationDf, EvaluationDf = None, None, None
        print("Invalid settings mode.")

    tr_df, _, _ = pu_label_process_trans(
        TrainingDf, CalibrationDf, EvaluationDf, num_lp, random_state
    )

    tr_df_concatnated = tr_df["title"] + " " + tr_df["abstract"]
    all_texts = tr_df_concatnated.tolist()
    vocab = build_vocab(all_texts)
    word_to_index = {word: index for index, word in enumerate(vocab)}
    all_features = getFeatures(tr_df, word_to_index, max_length=max_length)

    train_index = tr_df.query("tr == 1").index
    train_labels = tr_df.query("tr == 1")["pulabel"].values
    val_index = tr_df.query("ca == 1").index
    test_index = tr_df.query("ts == 1").index
    val_index = np.concatenate((val_index, test_index), axis=0)
    val_labels = tr_df.query("ca == 1")["label"].values
    test_labels_val = tr_df.query("ts == 1")["pulabel"].values
    val_labels = np.concatenate((val_labels, test_labels_val), axis=0)
    test_labels = tr_df.query("ts == 1")["label"].values
    train_data = BiDataset(
        torch.tensor(all_features)[train_index], torch.tensor(train_labels)
    )
    train_sampler = ProportionalSampler(train_data, batch_size=batch_size, num_cycles=1)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, drop_last=True
    )

    eval_dataset = BiDataset(torch.tensor(all_features)[val_index], val_labels)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = BiDataset(torch.tensor(all_features)[test_index], test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(len(vocab), embedding_dim).to(device)
    loss_fct = NonNegativePULoss(prior=prior)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs("PU_all_in_one/runs", exist_ok=True)
    writer = SummaryWriter(
        "PU_all_in_one/runs/covid_nnPU"
        + "_"
        + current_date
        + "_LP_from_"
        + str(settings_mode)
        + "_numLP:"
        + str(num_lp)
        + "_LR"
        + str(learning_rate)
    )

    best_va_f1 = -1
    best_ts_f1 = -1
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for i, (content, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            content = content.to(device)
            labels = labels.to(device)
            outputs = model(content[:, 0, :], content[:, 1, :])
            loss = loss_fct(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_prob = []
        with torch.no_grad():
            for i, (eval_content, _) in enumerate(eval_dataloader):
                eval_content = eval_content.to(device)
                eval_outputs = model(eval_content[:, 0, :], eval_content[:, 1, :])
                val_prob.append(eval_outputs.squeeze().cpu().numpy())
        val_prob = np.hstack(val_prob)
        npuu_eval_info_tuple = get_metric(labels=val_labels, prob=val_prob, mode="val")
        log_metrics(writer, "Validation", npuu_eval_info_tuple, epoch)
        npuu_val_threshold99 = npuu_eval_info_tuple[1]
        if npuu_eval_info_tuple[3] > best_va_f1:
            best_va_f1 = npuu_eval_info_tuple[3]

        test_prob = []
        with torch.no_grad():
            for i, (test_content, _) in enumerate(test_dataloader):
                test_content = test_content.to(device)
                test_outputs = model(test_content[:, 0, :], test_content[:, 1, :])
                test_prob.append(test_outputs.squeeze().cpu().numpy())
        test_prob = np.hstack(test_prob)
        npuu_test_info_tuple = get_metric(
            labels=test_labels,
            prob=test_prob,
            threshold99=npuu_val_threshold99,
            mode="test",
        )
        log_metrics(writer, "Test", npuu_test_info_tuple, epoch)
        if npuu_test_info_tuple[3] > best_ts_f1:
            best_ts_f1 = npuu_test_info_tuple[3]
            best_model_state = model.state_dict()
            torch.save(
                best_model_state,
                os.path.join(
                    covid_models,
                    f"npuu_model_f1_{best_ts_f1:.3f}_settings_mode_{settings_mode}_num_LP_{num_lp}.pth",
                ),
            )
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train nnPU model with CNN encoder on COVID-19 dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="PU_all_in_one/covid_task/Cochrane_COVID-19",
        help="Directory containing the COVID-19 dataset",
    )
    parser.add_argument(
        "--settings_mode",
        type=int,
        default=3,
        help="Settings mode for training nnPU model",
    )
    parser.add_argument(
        "--num_lp",
        type=int,
        default=50,
        help="Number of labeled positive samples",
    )
    parser.add_argument(
        "--random_state", type=int, default=1, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=256, help="Dimension of word embeddings"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum length of input sequences"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--prior", type=float, default=0.5, help="Prior for nnPU loss")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-8, help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--covid_models",
        type=str,
        default="saved_models/nnPU_CNN",
        help="Directory to save trained models",
    )
    args = parser.parse_args()
    train_nnpu_on_covid(
        data_dir=args.data_dir,
        settings_mode=args.settings_mode,
        num_lp=args.num_lp,
        random_state=args.random_state,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        batch_size=args.batch_size,
        prior=args.prior,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        covid_models=args.covid_models,
    )
    
    # python covid_nnpu_cnn.py --data_dir PU_all_in_one/covid_task/Cochrane_COVID-19 --settings_mode 3 --num_lp 50 --random_state 1 --embedding_dim 256 --max_length 512 --batch_size 32 --prior 0.5 --learning_rate 1e-8 --num_epochs 100 --covid_models PU_all_in_one/covid_task/Cochrane_COVID-19/ckpt
    # train_nnpu_on_covid(
    #     data_dir="PU_all_in_one/covid_task/Cochrane_COVID-19",
    #     settings_mode=3,
    #     num_lp=50,
    #     random_state=1,
    #     embedding_dim=256,
    #     max_length=512,
    #     batch_size=32,
    #     prior=0.5,
    #     learning_rate=1e-8,
    #     num_epochs=100,
    #     covid_models="PU_all_in_one/covid_task/Cochrane_COVID-19/ckpt",
    # )
