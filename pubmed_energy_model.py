import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
from data.pubmed_data_process import (
    make_PU_meta,
    BertDataset,
    ProportionalSampler,
)
from models.enegy_model import (
    EnergyNet,
    NonNegativePULoss,
    infer_prob,
    sample_langevin,
)
from utils import (
    set_seed,
    get_metric,
    log_metrics,
)


def train_em_on_pubmed(
    batch_size,
    num_epochs,
    experiment_list,
    prior,
    pubmed_models,
    seed,
    post_lr,
    prior_lr,
    cls_loss_weight,
    post_loss_weight,
    prior_loss_weight,
    runs_dir,
):
    set_seed(seed)
    bert_model_path = (
        r"/root/autodl-tmp/PU_all_in_one/pretrained/allenai/scibert_scivocab_uncased"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    experiments = [
        "/root/autodl-tmp/PU_all_in_one/data/pubmed-dse/L50/D000328.D008875.D015658",
        "/root/autodl-tmp/PU_all_in_one/data/pubmed-dse/L50/D000818.D001921.D051381",
        "/root/autodl-tmp/PU_all_in_one/data/pubmed-dse/L50/D006435.D007676.D008875",
        "/root/autodl-tmp/PU_all_in_one/data/pubmed-dse/L20/D000328.D008875.D015658",
        "/root/autodl-tmp/PU_all_in_one/data/pubmed-dse/L20/D000818.D001921.D051381",
        "/root/autodl-tmp/PU_all_in_one/data/pubmed-dse/L20/D006435.D007676.D008875",
    ]
    expriment_names = [
        "AMH_L50",
        "ABR_L50",
        "RKM_L50",
        "AMH_L20",
        "ABR_L20",
        "RKM_L20",
    ]

    for exper in experiment_list:
        root_dir = experiments[exper]

        tr_file_path = os.path.join(root_dir, "train.jsonl")
        va_file_path = os.path.join(root_dir, "valid.jsonl")
        ts_file_path = os.path.join(root_dir, "test.jsonl")

        all_df = make_PU_meta(tr_file_path, va_file_path, ts_file_path)
        train_labels = all_df.query("tr == 1")["pulabel"].values
        val_index = all_df.query("ca == 1").index
        val_labels = all_df.query("ca == 1")["pulabel"].values
        test_index = all_df.query("ts == 1").index
        test_labels = all_df.query("ts == 1")["label"].values

        tr_df_concatenated = all_df["title"] + " " + all_df["abstract"]
        tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
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
        bert_model = AutoModel.from_pretrained(bert_model_path).to(device)
        post_model = EnergyNet(768, 1).to(device)
        post_optimizer = torch.optim.Adamax(
            [p for p in post_model.parameters() if p.requires_grad is True], lr=post_lr
        )
        post_scheduler = CosineAnnealingLR(
            post_optimizer, T_max=num_epochs // 2, eta_min=0
        )
        prior_model = EnergyNet(768, 1).to(device)
        prior_optimizer = torch.optim.Adamax(
            [p for p in prior_model.parameters() if p.requires_grad is True],
            lr=prior_lr,
        )
        prior_scheduler = CosineAnnealingLR(
            prior_optimizer, T_max=num_epochs // 2, eta_min=0
        )
        loss_fn = NonNegativePULoss(prior)
        current_date = (
            datetime.now().strftime("%Y-%m-%d-%H-%M")
            + "_experiment_"
            + expriment_names[exper]
        )
        model_saved_path = os.path.join(os.path.join(pubmed_models, "EM"), current_date)
        os.makedirs(model_saved_path, exist_ok=True)
        os.makedirs(runs_dir, exist_ok=True)
        writer = SummaryWriter(
            runs_dir
            + "runs/pubmed_EM_SciBERT_"
            + current_date
            + "_experiment_"
            + expriment_names[exper]
        )
        best_test_auc = -1
        for epoch in tqdm(range(num_epochs)):
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
                        content_input_ids,
                        content_attention_mask,
                        content_token_type_ids,
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
                        + 0.1 * (pos_out_post**2 + neg_out_post**2)
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
                        + 0.1 * (pos_out_prior**2 + neg_out_prior**2)
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
                    eval_content_attention_mask = eval_content["attention_mask"].to(
                        device
                    )
                    eval_content_token_type_ids = eval_content["token_type_ids"].to(
                        device
                    )
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
                    test_content_attention_mask = test_content["attention_mask"].to(
                        device
                    )
                    test_content_token_type_ids = test_content["token_type_ids"].to(
                        device
                    )
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
    train_em_on_pubmed(
        batch_size=32,
        num_epochs=10,
        experiment_list=[1],
        prior=0.5,
        pubmed_models=r"/root/autodl-tmp/PU_all_in_one/saved_models/pubmed_models/",
        seed=42,
        post_lr=1e-4,
        prior_lr=1e-4,
        cls_loss_weight=1.0,
        post_loss_weight=0.9,
        prior_loss_weight=0.9,
        runs_dir="/root/autodl-tmp/PU_all_in_one/",
    )
