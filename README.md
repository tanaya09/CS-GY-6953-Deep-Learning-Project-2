# Parameter-Efficient AG News Classifier with LoRA Adapters

## Project Overview
This repository contains an implementation of a lightweight text classification model built on the AG News dataset using the Hugging Face Transformers library. Instead of fine-tuning all 125 million parameters of RoBERTa-base, we freeze the backbone and inject Low‑Rank Adaptation (LoRA) modules into the self‑attention layers. The result is a model with only **557 K trainable parameters** that converges rapidly and achieves **92.3 %** test accuracy in just three epochs.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Hyperparameter Choices](#hyperparameter-choices)
- [Lessons Learned](#lessons-learned)
- [License](#license)

## Features
- Parameter-efficient fine-tuning via LoRA adapters  
- Frozen RoBERTa-base encoder with only 0.4 % of weights trainable  
- Rapid convergence in 3 epochs  
- Drop-in inference compatibility with vanilla RoBERTa pipelines  

## Requirements
- Python 3.8+  
- transformers 
- torch  
- numpy  
- scikit-learn` 

## Installation
bash
https://github.com/tanaya09/CS-GY-6953-Deep-Learning-Project-2.git



Dataset & Preprocessing

Dataset: AG News (4 classes, 120 K train / 7.6 K test samples)
Tokenization: RobertaTokenizer with truncation+padding
Max length 256 (train), 128 (inference)
Augmentations: random token masking, synonym swaps
Model Architecture

Base: RoBERTa-base (12 layers, 768 hidden)
Adapters: LoRA modules injected into query & value projections
Trainable params: 557 954
Head: single linear classification layer + dropout
Training

Optimizer: AdamW
Learning rate: 1e-4
Warm-up: 10 %
Weight decay: 0.01
Epochs: 3
Batch size: 32
Results

Test Accuracy: 92.3 %
Training time: ~5 min/epoch on NVIDIA T4
Convergence: plateau by epoch 3
Inference: identical speed/memory to vanilla RoBERTa
