import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

from data.covid_data_process import (
    BertDataset,
    ProportionalSampler,
    parse_data,
    pu_label_process_trans,
    read_ris_file,
)

from models.enegy_model import (
    EnergyNet,
    NonNegativePULoss,
    infer_prob,
    sample_langevin,
)
from utils import (
    get_metric,
    log_metrics,
    set_seed,
)


def train_em_on_covid(
        data_dir,
        settings_mode,
        num_lp,
        random_state,
        batch_size,
        EPOCHS,
        post_lr,
        prior_lr,
        cls_loss_weight,
        post_loss_weight,
        prior_loss_weight,
        covid_models,
        runs_dir,
        bertmodel,
):
    set_seed(random_state)
    os.makedirs(covid_models, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(bertmodel)

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
    bert_model = AutoModel.from_pretrained(bertmodel).to(device)
    post_model = EnergyNet(768, 1).to(device)
    post_optimizer = torch.optim.Adamax(
        [p for p in post_model.parameters() if p.requires_grad is True], lr=post_lr
    )
    post_scheduler = CosineAnnealingLR(post_optimizer, T_max=EPOCHS // 2, eta_min=0)
    prior_model = EnergyNet(768, 1).to(device)
    prior_optimizer = torch.optim.Adamax(
        [p for p in prior_model.parameters() if p.requires_grad is True], lr=prior_lr
    )
    prior_scheduler = CosineAnnealingLR(prior_optimizer, T_max=EPOCHS // 2, eta_min=0)
    loss_fn = NonNegativePULoss(0.5)
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_saved_path = os.path.join(os.path.join(covid_models, "EM"), current_date)
    os.makedirs(model_saved_path, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(
        runs_dir
        + "runs/covid_EM_SciBERT"
        + "_"
        + current_date
        + "_LP_from_"
        + str(settings_mode)
        + "_numLP:"
        + str(num_lp)
    )
    best_test_auc = -1
    for epoch in tqdm(range(EPOCHS)):
        total_loss = 0.0
        for i, (content, train_batch_label) in enumerate(train_loader):
            post_model.train()
            prior_model.train()
            post_optimizer.zero_grad()
            prior_optimizer.zero_grad()
            content_input_ids = content["input_ids"].to(device)
            content_attention_mask = content["attention_mask"].to(device)
            content_token_type_ids = content["token_type_ids"].to(device)
            train_batch_label = train_batch_label.to(device)
            content_embedded = torch.mean(
                bert_model(
                    content_input_ids, content_attention_mask, content_token_type_ids
                ).last_hidden_state,
                dim=1,
            )

            train_batch_prob = infer_prob(
                post_model,
                prior_model,
                content_embedded,
            )
            train_batch_prob_positive = train_batch_prob[train_batch_label == 1]
            train_batch_prob_negative = train_batch_prob[train_batch_label == 0]
            cls_loss = loss_fn(train_batch_prob_positive, train_batch_prob_negative)

            train_batch_positive = content_embedded[train_batch_label == 1]
            if len(train_batch_positive) > 1:
                pos_post = train_batch_positive
                neg_post = torch.randn_like(pos_post)
                neg_post = sample_langevin(
                    neg_post, post_model, 0.01, 100, intermediate_samples=False
                ).to(device)
                pos_out_post = post_model(pos_post)
                neg_out_post = post_model(neg_post)
                loss_post = (
                        neg_out_post
                        - pos_out_post
                        + 0.1 * (pos_out_post ** 2 + neg_out_post ** 2)
                ).mean()
            else:
                loss_post = torch.tensor([0], dtype=torch.float32).to(device)

            train_batch_unlabeled = content_embedded[train_batch_label == 0]
            if len(train_batch_unlabeled) > 1:
                pos_prior = train_batch_unlabeled
                neg_prior = torch.randn_like(pos_prior)
                neg_prior = sample_langevin(
                    neg_prior, prior_model, 0.01, 100, intermediate_samples=False
                ).to(device)
                pos_out_prior = prior_model(pos_prior)
                neg_out_prior = prior_model(neg_prior)
                loss_prior = (
                        neg_out_prior
                        - pos_out_prior
                        + 0.1 * (pos_out_prior ** 2 + neg_out_prior ** 2)
                ).mean()
            else:
                loss_prior = torch.tensor([0], dtype=torch.float32).to(device)

            los_sum = (
                    cls_loss_weight * cls_loss
                    + post_loss_weight * loss_post
                    + prior_loss_weight * loss_prior
            )
            los_sum.backward()
            torch.nn.utils.clip_grad_norm_(post_model.parameters(), max_norm=0.1)
            torch.nn.utils.clip_grad_norm_(prior_model.parameters(), max_norm=0.1)
            post_optimizer.step()
            prior_optimizer.step()
            post_scheduler.step()
            prior_scheduler.step()
            total_loss += los_sum.item()
        print(f"Epoch {epoch} Loss: {total_loss}")

        val_prob = []
        post_model.eval()
        prior_model.eval()
        with torch.no_grad():
            for i, (eval_content, _) in enumerate(eval_dataloader):
                eval_content_input_ids = eval_content["input_ids"].to(device)
                eval_content_attention_mask = eval_content["attention_mask"].to(device)
                eval_content_token_type_ids = eval_content["token_type_ids"].to(device)
                eval_embedded = torch.mean(
                    bert_model(
                        eval_content_input_ids,
                        eval_content_attention_mask,
                        eval_content_token_type_ids,
                    ).last_hidden_state,
                    dim=1,
                )
                test_batch_prob = infer_prob(
                    post_model,
                    prior_model,
                    eval_embedded,
                )
                val_prob.append(test_batch_prob.cpu())
        val_prob = torch.hstack(val_prob).numpy()
        em_val_info_tuple = get_metric(val_labels, val_prob)
        log_metrics(writer, "Validation", em_val_info_tuple, epoch)
        em_val_threshold99 = em_val_info_tuple[1]

        test_prob = []
        with torch.no_grad():
            for i, (test_content, _) in enumerate(test_dataloader):
                test_content_input_ids = test_content["input_ids"].to(device)
                test_content_attention_mask = test_content["attention_mask"].to(device)
                test_content_token_type_ids = test_content["token_type_ids"].to(device)
                test_embbeded = torch.mean(
                    bert_model(
                        test_content_input_ids,
                        test_content_attention_mask,
                        test_content_token_type_ids,
                    ).last_hidden_state,
                    dim=1,
                )
                test_batch_prob = infer_prob(
                    post_model,
                    prior_model,
                    test_embbeded,
                )
                test_prob.append(test_batch_prob.cpu())
        test_prob = torch.hstack(test_prob).numpy()
        em_test_info_tuple = get_metric(test_labels, test_prob)
        log_metrics(writer, "Test", em_test_info_tuple, epoch)
        if em_test_info_tuple[3] > best_test_auc:
            best_test_auc = em_test_info_tuple[3]

            torch.save(
                post_model.state_dict(),
                os.path.join(
                    model_saved_path,
                    f"post_model_best_test_auc_epoch_{epoch}_auc_{best_test_auc:.3f}.pth",
                ),
            )
            torch.save(
                prior_model.state_dict(),
                os.path.join(
                    model_saved_path,
                    f"prior_model_best_test_auc_epoch_{epoch}_auc_{best_test_auc:.3f}.pth",
                ),
            )
    writer.close()


if __name__ == "__main__":
    train_em_on_covid(
        data_dir=r"/root/autodl-tmp/PU_all_in_one/data/Cochrane_Covid-19",
        settings_mode=3,
        num_lp=50,
        random_state=42,
        batch_size=32,
        EPOCHS=30,
        post_lr=5e-5,
        prior_lr=5e-5,
        cls_loss_weight=1,
        post_loss_weight=0.9,
        prior_loss_weight=0.9,
        covid_models=r"/root/autodl-tmp/PU_all_in_one/saved_models",
        runs_dir=r"/root/autodl-tmp/PU_all_in_one/",
        bertmodel=r"/root/autodl-tmp/PU_all_in_one/pretrained/allenai/scibert_scivocab_uncased",
    )
