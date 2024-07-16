import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn as nn
import copy
from intermediate_fusion_brute_force_search import new_train_intermediate, new_validate_intermediate
from intermediate_fusion_brute_force_search import get_fusion_model_and_dataloaders, load_model

#-------------------------------------------Definition of functions---------------------------------------------------------
#Simulated Annealing Algorithm
def SAA(objf, initial_solution, lb, ub, max_iter, initial_temp, cooling_rate):
    # Initial solution and objective value
    current_solution = initial_solution
    current_value = objf(current_solution)

    # Best solution and objective value
    best_solution = copy.deepcopy(current_solution)
    best_value = current_value

    # Initial temperature
    temp = initial_temp

    for iter in range(max_iter):
        # Generate a new candidate solution by perturbing the current solution
        new_solution = current_solution + np.random.uniform(-1.5, 1.5, size=initial_solution.shape)

        # Ensure the new solution is within the bounds
        new_solution = np.clip(new_solution, lb, ub)

        # Evaluate the new solution
        new_value = objf(new_solution)

        # Calculate the change in objective value
        delta_value = new_value - current_value

        # Acceptance criterion
        if delta_value < 0 or np.exp(-delta_value / temp) > random.random():
            current_solution = new_solution
            current_value = new_value

            # Update the best solution if the new solution is better
            if new_value < best_value:
                best_solution = copy.deepcopy(new_solution)
                best_value = new_value
                # final_checkpoint = torch.load('temp_SAA_best_model_min_val_loss.pth')
                # torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'SAA_best_model.pth')

        # Cool down the temperature
        temp *= cooling_rate

    best_model_save = objf(best_solution)
    final_checkpoint = torch.load('temp_SAA_best_model_min_val_loss.pth')
    torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'SAA_best_model.pth')

    return best_solution, best_value


def fitness_function_factory_SAA(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion):
    def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion):
        #create intermediate fusion head for fused MLP models
        model, train_loaders, val_loaders, _ = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution, mode, device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        model_path = 'temp_SAA_best_model_min_val_loss.pth'
        #train and validate fused MLP models with fusion head
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']                                               #Validation results of intermediate_fusion_SAA
        return loss

    def fitness_func_SAA(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion)
        return solution_fitness
    return fitness_func_SAA

#------------------------------------------------- SAA optimization----------------------------------------------------------
def intermediate_fusion_SAA(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, mode, criterion):
    fitness_func_SAA = fitness_function_factory_SAA(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion)
    # Calculate number of fused MLP models + fusion head where to optimize the number of NN layers
    dim = sum(1 for data_type in dimension_dict.keys() for i in loaders_dict["train"][data_type])
    lb = np.array([1] * dim)                                                    # Lower bounds for the number of layers
    ub = [int(np.log2(up_b)) for up_b in dimension_dict.values()]               #  Upper bounds for the number of layers
    initial_solution = np.random.uniform(lb, ub)                                # Initial solution (starting point)
    initial_temp = 100.0                                                        # Initial temperature
    cooling_rate = 0.95                                                         # Cooling rate
    best_solution, solution_fitness = SAA(fitness_func_SAA, initial_solution, lb, ub, max_iter, initial_temp, cooling_rate)   # return the best combination of NN layers and its loss
    os.remove('temp_SAA_best_model_min_val_loss.pth')
    model, _, _, test_loaders = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, np.around(best_solution), mode, device)
    test_model = load_model(model, 'SAA_best_model.pth')
    test_loss = new_validate_intermediate(test_model, test_loaders, criterion, device)     #test model with the best combination of layers
    return np.around(best_solution).astype(int), test_loss
