import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
from data.covid_data_process import (
    BertDataset,
    ProportionalSampler,
    parse_data,
    pu_label_process_trans,
    read_ris_file,
)
from models.nnPU import NonNegativePULoss, PUModelWithSciBERT
from utils import (
    get_metric,
    log_metrics,
    set_seed,
)


def train_nnpu_on_covid(
        data_dir,
        settings_mode,
        num_lp,
        random_state,
        batch_size,
        prior,
        learning_rate,
        num_epochs,
        covid_models,
        runs_dir,
):
    set_seed(random_state)
    os.makedirs(covid_models, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    Bertmodel = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    Bertmodel = Bertmodel.to(device)

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
    train_index = tr_df.query("tr == 1").index
    train_labels = tr_df.query("tr == 1")["pulabel"].values
    val_index = tr_df.query("ca == 1").index
    test_index = tr_df.query("ts == 1").index
    val_index = np.concatenate((val_index, test_index), axis=0)
    val_labels = tr_df.query("ca == 1")["label"].values
    test_labels_val = tr_df.query("ts == 1")["pulabel"].values
    val_labels = np.concatenate((val_labels, test_labels_val), axis=0)
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
    train_sampler = ProportionalSampler(
        train_dataset, batch_size=batch_size, num_cycles=1
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
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

    test_inputs = tokenizer(
        tr_df_concatenated[test_index].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    test_dataset = BertDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PUModelWithSciBERT(model=Bertmodel).to(device)
    loss_fct = NonNegativePULoss(prior=prior)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(
        runs_dir
        + "runs/covid_nnPU_SciBERT"
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
            content_input_ids = content["input_ids"].to(device)
            content_attention_mask = content["attention_mask"].to(device)
            content_token_type_ids = content["token_type_ids"].to(device)
            labels = labels.to(device)
            outputs = model(
                content_input_ids, content_attention_mask, content_token_type_ids
            )
            loss = loss_fct(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_prob = []
        with torch.no_grad():
            for i, (eval_content, _) in enumerate(eval_dataloader):
                eval_content_input_ids = eval_content["input_ids"].to(device)
                eval_content_attention_mask = eval_content["attention_mask"].to(device)
                eval_content_token_type_ids = eval_content["token_type_ids"].to(device)
                eval_outputs = model(
                    eval_content_input_ids,
                    eval_content_attention_mask,
                    eval_content_token_type_ids,
                )
                val_prob.append(eval_outputs.squeeze().cpu().numpy())
        val_prob = np.hstack(val_prob)
        npuu_eval_info_tuple = get_metric(labels=val_labels, prob=val_prob)
        log_metrics(writer, "Validation", npuu_eval_info_tuple, epoch)
        npuu_val_threshold99 = npuu_eval_info_tuple[1]
        if npuu_eval_info_tuple[3] > best_va_f1:
            best_va_f1 = npuu_eval_info_tuple[3]

        test_prob = []
        with torch.no_grad():
            for i, (test_content, _) in enumerate(test_dataloader):
                test_content_input_ids = test_content["input_ids"].to(device)
                test_content_attention_mask = test_content["attention_mask"].to(device)
                test_content_token_type_ids = test_content["token_type_ids"].to(device)
                test_outputs = model(
                    test_content_input_ids,
                    test_content_attention_mask,
                    test_content_token_type_ids,
                )
                test_prob.append(test_outputs.squeeze().cpu().numpy())
        test_prob = np.hstack(test_prob)
        npuu_test_info_tuple = get_metric(
            labels=test_labels,
            prob=test_prob,
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
    data_dir = r"/home/dudu/all/PU_all_in_one/data/Cochrane_COVID-19"
    settings_mode = 3
    num_lp = 50
    random_state = 42
    batch_size = 24
    prior = 0.5
    learning_rate = 3e-6
    num_epochs = 10
    covid_models = r"/home/dudu/all/PU_all_in_one/covid_task/saved_models/nnPU/"
    runs_dir = r"/home/dudu/all/PU_all_in_one/"
    train_nnpu_on_covid(
        data_dir,
        settings_mode,
        num_lp,
        random_state,
        batch_size,
        prior,
        learning_rate,
        num_epochs,
        covid_models,
        runs_dir,
    )
