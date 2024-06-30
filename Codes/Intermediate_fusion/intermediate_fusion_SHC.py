import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn as nn
import copy
from early_fusion import MLP
from intermediate_fusion_brute_force_search import NewMyEnsemble, new_train_intermediate

#-------------------------------------------Definition of functions---------------------------------------------------------
#Simple Hill Climbing
def generate_neighbors(solution, step_size=0.1):
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution.copy()
        neighbor[i] += random.uniform(-step_size, step_size)
        neighbors.append(neighbor)
    return neighbors

def SHC(objf, lb, ub, dim, max_iter, step_size=0.1):
    current_solution = np.random.uniform(lb, ub, size=dim)
    current_fitness = objf(current_solution)

    for i in range(max_iter):
        neighbors = generate_neighbors(current_solution, step_size)
        best_neighbor = None
        best_fitness = current_fitness

        for neighbor in neighbors:
            fitness = objf(neighbor)
            if fitness < best_fitness:
                best_neighbor = neighbor
                best_fitness = fitness

        if best_fitness < current_fitness:
            current_solution = best_neighbor
            current_fitness = best_fitness

    return current_solution, current_fitness

def fitness_function_factory_SHC(dimension_dict, loaders_dict, device, lr, num_epochs, criterion):
    def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion):
        train_loaders, val_loaders = [], []
        models = []
        index = 0
        for data_type in loaders_dict.keys():
            train_loaders_per_type, val_loaders_per_type = [], []
            for i, (train_loader, val_loader) in enumerate(loaders_dict[data_type]):
                train_loaders_per_type.append(train_loader)
                val_loaders_per_type.append(val_loader)
                input_dim = dimension_dict[data_type]
                model = MLP(input_dim=input_dim, n_layers=round(solution[index]), fusion="intermediate")
                models.append(model)
                index += 1
            train_loaders.extend(train_loaders_per_type)
            val_loaders.extend(val_loaders_per_type)
        model = NewMyEnsemble(models, n_layers=round(solution[index]))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #Training
        model_path = 'SHC_best_model_min_val_loss.pth'
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']
        return loss

    def fitness_func_SHC(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
        return solution_fitness
    return fitness_func_SHC

#------------------------------------------------- SHC optimization----------------------------------------------------------
def intermediate_fusion_SHC(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, criterion):
    fitness_func_SHC = fitness_function_factory_SHC(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    dim = sum(1 for data_type in loaders_dict.keys() for i in loaders_dict[data_type]) + 1         # Number of solutions
    lb, ub = 1, 10 #  Lower and upper bound of the search space
    step_size = 0.1  # Step size for the search space
    solution, solution_fitness = SHC(fitness_func_SHC, lb, ub, dim, max_iter, step_size)
    return np.around(solution), solution_fitness