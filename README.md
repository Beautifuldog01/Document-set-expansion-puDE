# Document-set-expansion-puDE

This repository contains the code for the paper ["Document Set Expansion with Positive-Unlabelled Learning Using Intractable Density Estimation" (COLING 2024)](https://arxiv.org/pdf/2403.17473.pdf).

<img width="834" alt="image" src="https://github.com/Beautifuldog01/Document-set-expansion-puDE/assets/40363660/bf505cd5-a124-40d6-a4bc-6eb571ca4bf5">

## Requirements

The code is written in Python 3.10. To install the required packages, you need have [Conda](https://docs.anaconda.com/free/miniconda/) installed. You can then create a new environment with the required packages using the following command:

```zsh
conda create --name pypu python=3.10 --file requirements.txt
conda activate pypu
```

To git clone the repository, you can use the following command:

```zsh
git clone https://github.com/Beautifuldog01/Document-set-expansion-puDE.git
```

After you have cloned the repository, you will get the following directory structure:

```zsh
├── LICENSE
├── README.md
├── covid_MLT.py
├── covid_energy_model.py
├── covid_nnpu.py
├── covid_nnpu_cnn.py
├── covid_vpu.py
├── data
│   ├── Cochrane_Covid-19
│   ├── covid_data_process.py
│   ├── pubmed-dse
│   └── pubmed_data_process.py
├── models
│   ├── enegy_model.py
│   ├── nnPU.py
│   ├── nnPU_cnn.py
│   └── vpu.py
├── pubmed_energy_model.py
└── utils.py
```

The code contains four models: `vpu`, `nnPU`, `nnPU_cnn`, and `energy_model`. The implementation of `vpu` in this repository is largely based on the work found in [A Variational Approach for Learning from Positive and Unlabeled Data](https://github.com/HC-Feynman/vpu). The `nnPU` is basically from [nnPUlearning](https://github.com/kiryor/nnPUlearning), and the `nnPU_cnn` model is my previous work in [AcademicDocumentClassifier_without_AllenNLP](https://github.com/Beautifuldog01/AcademicDocumentClassifier_without_AllenNLP), and the `energy_model` model is the energy-based PU learning model. The `covid_vpu.py`, `covid_nnpu.py`, `covid_nnpu_cnn.py`, and `covid_energy_model.py` files contain the code for the `vpu`, `nnPU`, `nnPU_cnn`, and `energy_model` models, respectively. The `covid_data_process.py` file contains the code for processing the COVID-19 dataset, and the `pubmed_data_process.py` file contains the code for processing the PubMed dataset.

## Usage

We use tensorboard to visualize the training process. To start tensorboard, you can use the following command:

```zsh
tensorboard --logdir runs/
```

### nnPU

To train the `nnPU` on the COVID-19 dataset, you can use the following command:

```zsh
python covid_nnpu.py \
    --data_dir data/Cochrane_Covid-19 \
    --settings_mode 3 \
    --num_lp 50 \
    --random_state 42 \
    --batch_size 24 \
    --prior 0.5 \
    --learning_rate 3e-6 \
    --num_epochs 10 \
    --covid_models saved_models/nnPU/ \
    --runs_dir runs/ \
    --bertmodel allenai/scibert_scivocab_uncased
```

or with the `CNN encoder`, you can use the following command:

```zsh
python covid_nnpu_cnn.py \
    --data_dir data/Cochrane_Covid-19 \
    --settings_mode 3 \
    --num_lp 50 \
    --random_state 1 \
    --embedding_dim 256 \
    --max_length 512 \
    --batch_size 32 \
    --prior 0.5 \
    --learning_rate 1e-8 \
    --num_epochs 100 \
    --covid_models saved_models/nnPU_cnn/
```

### vpu

To train the `vpu` on the COVID-19 dataset, you can use the following command:

```zsh
python covid_vpu.py \
    --data_dir data/Cochrane_Covid-19 \
    --settings_mode 3 \
    --num_lp 50 \
    --random_state 42 \
    --batch_size 32 \
    --bertmodelpath allenai/scibert_scivocab_uncased \
    --filepath saved_models/vpu/vpu.pth \
    --learning_rate 3e-5 \
    --lam 0.1 \
    --mix_alpha 0.1 \
    --epochs 10 \
    --val_iterations 20
```

### Energy Model

To train the `energy model` on the COVID-19 dataset, you can use the following command:

```zsh
 python covid_energy_model.py \
    --data_dir data/Cochrane_Covid-19 \
    --settings_mode 3 \
    --num_lp 50 \
    --random_state 42 \
    --batch_size 32 \
    --EPOCHS 30 \
    --post_lr 5e-5 \
    --prior_lr 5e-5 \
    --cls_loss_weight 1 \
    --post_loss_weight 0.9 \
    --prior_loss_weight 0.9 \
    --covid_models saved_models/EM_covid/ \
    --runs_dir runs/ \
    --bertmodel allenai/scibert_scivocab_uncased
```

To train the `energy model` on the PubMed dataset, you can use the following command:

You can change the `experiment_list` from `0` to `5` which is the index of the experiment_names list.

```python
expriment_names = [
    "AMH_L50",
    "ABR_L50",
    "RKM_L50",
    "AMH_L20",
    "ABR_L20",
    "RKM_L20",
]
```

Here, `AMH_L50` means the experiment with the `Adult+Middle Aged+HIV infections` dataset and the number of the labeled postivie samples is `50`. The same goes for the other experiment names.

```zsh
python pubmed_energy_model.py \
--batch_size 32 \
--num_epochs 10 \
--experiment_list 1 \
--prior 0.5 \
--pubmed_models saved_models/EM_pubmed/ \
--seed 42 \
--post_lr 1e-4 \
--prior_lr 1e-4 \
--cls_loss_weight 1.0 \
--post_loss_weight 0.9 \
--prior_loss_weight 0.9 \
--runs_dir runs/ \
--bert_model_path allenai/scibert_scivocab_uncased \
--data_dir data/pubmed-dse
```

## Citation

Please cite our paper if you use this code in your own work:

```bibtex
@misc{zhang2024document,
      title={Document Set Expansion with Positive-Unlabelled Learning Using Intractable Density Estimation}, 
      author={Haiyang Zhang and Qiuyi Chen and Yuanjie Zou and Yushan Pan and Jia Wang and Mark Stevenson},
      year={2024},
      eprint={2403.17473},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
