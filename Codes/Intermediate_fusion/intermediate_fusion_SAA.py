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
        new_solution = current_solution + np.random.uniform(-1.5, 1.5, size=initial_solution.shape)  #we can change the limits of stepsize (-1.5, 1.5)

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

        # Cool down the temperature
        temp *= cooling_rate

    return best_solution, best_value


def fitness_function_factory_SAA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion):
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
        model_path = 'SAA_best_model_min_val_loss.pth'
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']
        return loss

    def fitness_func_SAA(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
        return solution_fitness
    return fitness_func_SAA

#------------------------------------------------- SAA optimization----------------------------------------------------------
def intermediate_fusion_SAA(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, criterion):
    fitness_func_SAA = fitness_function_factory_SAA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    dim = sum(1 for data_type in loaders_dict.keys() for i in loaders_dict[data_type]) + 1         # Number of solutions
    lb = np.array([1] * dim)  # Lower bounds for the dimensions
    ub = np.array([10] * dim)  # Upper bounds for the dimensions
    initial_solution = np.array([5] * dim)  # Initial solution (starting point)
    initial_temp = 100.0   #Initial temperature
    cooling_rate = 0.95    # Cooling rate
    solution, solution_fitness = SAA(fitness_func_SAA, initial_solution, lb, ub, max_iter, initial_temp, cooling_rate)
    return np.around(solution).astype(int), solution_fitness
