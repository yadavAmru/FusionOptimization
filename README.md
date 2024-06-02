# Fusion Optimization

## Overview

In this project, we have developed and implemented three fusion models: Early fusion, Late fusion, and Intermediate fusion. These models are designed to predict outputs from multimodal data, and we aim to compare their performance to determine which fusion strategy yields the most accurate results. Our primary focus is on intermediate fusion, which often provides better results than early and late fusion. Additionally, we have applied optimization techniques, namely Brute Force, Grey Wolf Optimizer (GWO), Particle Swarm Optimization (PSO),  and Genetic Algorithms (GA), to the intermediate fusion model to further enhance its performance.

## Installation

To start this project, clone the repository and install the necessary dependencies.
```bash
git clone https://github.com/yadavAmru/FusionOptimization

## Usage

--write later
To load the dataset train the fusion models  and evaluate the fusion models, run the Dataset.py and main.py scripts. This script loads the trains the early, late, and intermediate fusion models and compares their performance.
'''bash
python main.py

## Fusion Models

![Model Diagram](image_path)
Installation: <br/>
The **"Codes" folder** contains Python files related to Fusion Optimization. <br/>
**Damir Kanymkulov** created the following Python files: <br/>
- The **"Dataset.py" file** contains a code which loads, preprocesses the dataset with different data types (images, numerical data), and creates dataloaders that are convenient for training neural networks.
- The **"brute_force_search.py" file** includes a code that performs an intermediate fusion of two MLP models with different data types. The process of finding the best fusion location was done using a brute-force search algorithm.
- The **"intermediate_fusion.py" file** has a code that performs an intermediate fusion of two MLP models with different data types. The process of finding the best fusion location was optimized using a Genetic Algorithm (GA).
- The **"interm_fusion_GWO.py" file** has a code that performs an intermediate fusion of two MLP models with different data types. The process of finding the best fusion location was optimized using Grey Wolf Optimization (GWO).
- The **"interm_fusion_PSO.py" file** has a code that performs an intermediate fusion of two MLP models with different data types. The process of finding the best fusion location was optimized using Particle Swarm Optimization (PSO).
- The **"late_fusion.py" file** contains a code that performs training of two MLP models with different data types and returns an average of two results (late fusion).
- The **"main.py" file** includes the main part of the fusion optimization framework that defines hyperparameters, runs all the fusion functions, and prints their results.

**Amruta Yadav** created the following Python files: <br/>
- The **"early_fusion.py" file** contains a code that performs training of early fusion. Early fusion includes concatenating two different data types together and then training one MLP model using this combined dataset.
