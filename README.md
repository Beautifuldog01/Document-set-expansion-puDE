# Document-set-expansion-puDE

This repository contains the code for the paper ["Document Set Expansion with Positive-Unlabelled Learning Using Intractable Density Estimation" (COLING 2024)](https://arxiv.org/pdf/2403.17473.pdf).

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
