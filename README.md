# assignment4-autoencoders-rbm-dbn
# Autoencoders, Restricted Boltzmann Machines, and Deep Belief Networks

## Overview

This project explores unsupervised representation learning and reconstruction using the Scikit-learn Digits dataset. It investigates autoencoders, sparse and denoising autoencoders, variational autoencoders (VAEs), Restricted Boltzmann Machines (RBMs), and RBM-based layer-wise pretraining for classification.

## Objectives

* Implement an undercomplete autoencoder with a two-dimensional latent space.
* Compare learned latent representations with PCA.
* Investigate sparsity regularization and its effect on reconstruction.
* Evaluate denoising performance at different noise levels.
* Explore latent representations using a variational autoencoder.
* Train an RBM using one-step Contrastive Divergence (CD-1).
* Compare RBM-pretrained classification against random initialization.

## Dataset

The experiments use the Scikit-learn Digits dataset, containing 1,797 handwritten digit images across ten classes. Each image is represented by 64 features corresponding to an 8 × 8 image.

Pixel values are normalized by dividing by 16. The dataset is split into 80% training data and 20% testing data, using stratification and `random_state=42`.

## Technologies

* Python
* PyTorch
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Reported Results

| Experiment                           |       Reported result |
| ------------------------------------ | --------------------: |
| Sparse autoencoder, β = 0.0          |     Test MSE: 0.02838 |
| Sparse autoencoder, β = 0.1          |     Test MSE: 0.07325 |
| Denoising autoencoder, noise σ = 0.6 |    Test MSE: 0.066259 |
| Variational autoencoder              |    Test MSE: 0.074090 |
| Restricted Boltzmann Machine         |    Test MSE: 0.073893 |
| RBM-pretrained classifier            | Test accuracy: 79.44% |
| Randomly initialized classifier      | Test accuracy: 81.39% |

These metrics are transcribed from the accompanying project report and have not been independently rerun here. In the reported classifier experiment, random initialization outperformed RBM pretraining by 1.95 percentage points.

## Project Structure

```text
assignment4-autoencoders-rbm-dbn/
├── notebooks/
│   └── assignment4p12.ipynb
├── reports/
│   └── Detailed_Project_Report_Autoencoders_RBM_DBN.docx
├── docs/
│   └── experiment-notes.md
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Installation

Create and activate a Python virtual environment.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the notebook:

```bash
jupyter notebook notebooks/assignment4p12.ipynb
```

Run the notebook from top to bottom to reproduce the experiments and visualizations.

## Limitations and Future Work

* Run multiple trials and report mean and standard deviation.
* Use a validation set for model selection.
* Report numerical test reconstruction error for the undercomplete autoencoder.
* Improve sparse-activation visualizations.
* Record software versions and experiment configurations.

## Academic Integrity

Credit the dataset and any external sources used. Clearly identify adapted code and distinguish reproduced results from new experiments.
