<h1 align= "center"> Fusion Optimization </h1>

## Overview

In this project, we have developed and implemented three fusion models: Early fusion, Late fusion, and Intermediate fusion. These models are designed to predict outputs from multimodal data, and we aim to compare their performance to determine which fusion strategy yields the most accurate results. Our primary focus is on intermediate fusion, which often provides better results than early and late fusion. Additionally, we have applied optimization techniques, namely Brute Force, Grey Wolf Optimizer (GWO), Particle Swarm Optimization (PSO),  and Genetic Algorithms (GA), to the intermediate fusion model to further enhance its performance.

## Installation

To start this project, clone the repository and install the necessary dependencies.
```bash
!git clone https://github.com/yadavAmru/FusionOptimization
```

## Usage

To load and preprocess the dataset execute **"Dataset.py"** file and to execute all the fusion functions, calculate and compare the results execute **"main.py"** file.


## Fusion Models

![Fusion Diagram](https://github.com/yadavAmru/FusionOptimization/blob/main/Images/Fusion_Diagrams.png)

### A. Early Fusion:
In early fusion, the data from multiple inputs is concatenated at the input level, passed through the single model, and then the output is predicted. This approach allows the model to learn from the combined data directly. 
- The **"early_fusion.py"** file has a code that performs early fusion.

### B. Late Fusion:
In late fusion, separate models are trained for each data input and the output for each model is predicted. The output of all the models is then aggregated at the decision level to calculate the final outcome. 
- The **"late_fusion.py"** file has a code that performs late fusion.

### C. Intermediate Fusion:
In intermediate fusion, features are extracted separately from each input data using specialized models. These features are then fused at an intermediate layer, and the combined features are processed jointly to make the final prediction.
- The **"intermediate_fusion.py"** file has a code that performs an intermediate fusion of two MLP models with different data types. The process of finding the best fusion location was optimized using a Genetic Algorithm (GA).
- The **"interm_fusion_GWO.py"** file has a code that performs an intermediate fusion of two MLP models with different data types. The process of finding the best fusion location was optimized using Grey Wolf Optimization (GWO).
- The **"interm_fusion_PSO.py"** file has a code that performs an intermediate fusion of two MLP models with different data types. The process of finding the best fusion location was optimized using Particle Swarm Optimization (PSO).
- The **"brute_force_search.py"** file includes a code that performs an intermediate fusion of two MLP models with different data types. The process of finding the best fusion location was done using a brute-force search algorithm.
