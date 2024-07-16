import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from intermediate_fusion_brute_force_search import new_train_intermediate, new_validate_intermediate
from intermediate_fusion_brute_force_search import get_fusion_model_and_dataloaders, load_model

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

    best_model_save = objf(current_solution)
    final_checkpoint = torch.load('temp_SHC_best_model_min_val_loss.pth')
    torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'SHC_best_model.pth')

    return current_solution, current_fitness


def fitness_function_factory_SHC(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion):
    def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion):
        #create intermediate fusion head for fused MLP models
        model, train_loaders, val_loaders, _ = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution, mode, device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        model_path = 'temp_SHC_best_model_min_val_loss.pth'
        #train and validate fused MLP models with fusion head
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']                                               #Validation results of intermediate_fusion_SAA
        return loss

    def fitness_func_SHC(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion)
        return solution_fitness
    return fitness_func_SHC

#------------------------------------------------- SHC optimization----------------------------------------------------------
def intermediate_fusion_SHC(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, mode, criterion):
    fitness_func_SHC = fitness_function_factory_SHC(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion)
    # Calculate number of fused MLP models + fusion head where to optimize the number of NN layers
    dim = sum(1 for data_type in dimension_dict.keys() for i in loaders_dict["train"][data_type])             # Number of solutions
    lb, ub = 1, [int(np.log2(up_b)) for up_b in dimension_dict.values()]                                           #  Lower and upper bound of the search space
    step_size = 0.5                                                                                           # Step size for the search space

    best_solution, solution_fitness = SHC(fitness_func_SHC, lb, ub, dim, max_iter, step_size)  # return the best combination of NN layers and its loss
    os.remove('temp_SHC_best_model_min_val_loss.pth')
    model, _, _, test_loaders = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, np.around(best_solution), mode, device)
    test_model = load_model(model, 'SHC_best_model.pth')
    test_loss = new_validate_intermediate(test_model, test_loaders, criterion, device)        #test model with the best combination of layers
    return np.around(best_solution).astype(int), test_loss
