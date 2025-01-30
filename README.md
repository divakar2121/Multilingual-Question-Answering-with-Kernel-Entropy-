# Multilingual-Question-Answering-with-Kernel-Entropy-



This project demonstrates how to evaluate the performance of transformer models on multilingual question-answering tasks using kernel entropy as a measure of uncertainty. The project includes models such as `DistilBERT`, `BERT`, `RoBERTa`, and `ALBERT`, and evaluates their performance using AUROC scores.

## Table of Contents

- [Overview](#overview)
- [Models Used](#models-used)
- [Setup and Installation](#setup-and-installation)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Overview

This project leverages different transformer models to predict uncertainty in question answering tasks. The uncertainty is computed using kernel entropy, and the model performance is evaluated using AUROC (Area Under the Receiver Operating Characteristic Curve). The dataset includes multilingual questions from languages like Tamil, English, Hindi, German, Japanese, French, Italian, and Spanish.

## Models Used

This project uses the following pre-trained transformer models:

- **DistilBERT**
- **BERT**
- **RoBERTa**
- **ALBERT**

The performance of these models is compared using kernel entropy for uncertainty estimation.


    ```

## Dataset

The dataset consists of multilingual question-answer pairs, with the following languages included:

- Tamil
- English
- Hindi
- German
- Japanese
- French
- Italian
- Spanish

Each question is labeled with `1` for correct answers and `0` for incorrect answers.

## Evaluation

The evaluation is based on calculating the **kernel entropy** for each answer using embeddings from the transformer models. The model's uncertainty is quantified, and the performance is assessed using **AUROC** scores.

## Results

The results are displayed as AUROC scores for each transformer model on the multilingual question-answer dataset. These scores provide insight into how well each model handles uncertainty estimation for multilingual questions.

**Example output:**

```text
Results for DistilBERT:
  AUROC for Multilingual dataset: 0.65625
==================================================
Results for BERT:
  AUROC for Multilingual dataset: 0.38541666666666663
==================================================
Results for RoBERTa:
  AUROC for Multilingual dataset: 0.5520833333333333
==================================================
Results for ALBERT:
  AUROC for Multilingual dataset: 0.5625
==================================================
