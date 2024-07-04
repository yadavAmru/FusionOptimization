import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy as np
from intermediate_fusion_brute_force_search import new_train_intermediate, new_validate_intermediate
from intermediate_fusion_brute_force_search import get_fusion_model_and_dataloaders, load_model

#-------------------------------------------Definition of functions---------------------------------------------------------
def SMA(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # Initialize the positions of slime mould
    positions = np.random.uniform(lb, ub, size=(SearchAgents_no, dim))
    Best_pos = None
    Best_score = float("inf")
    w = 0.9  #  Weight parameter
    vb = 0   #  Vibration parameter
    z  = 0.1 #  Random parameter
    for i in range(Max_iter):
        # Calculate the fitness value for each agent
        fitness_value = np.array([objf(pos) for pos in positions])

        # Sort the agents based on fitness value from best to worst
        sorted_indices = np.argsort(fitness_value)
        positions = positions[sorted_indices]

        # Update the weights
        w = w * np.exp(-i / Max_iter)

        # Update the positions using the SMA equation
        for j in range(SearchAgents_no):
            if random.random() < w:
                # Approach food (exploitation)
                Best_pos = positions[0]
                positions[j] = positions[j] + np.random.rand() * (Best_pos - positions[j])
            else:
                # Explore randomly
                random_index = random.randint(0, SearchAgents_no - 1)
                random_pos = positions[random_index]
                positions[j] = positions[j] + z * np.random.rand() * (random_pos - positions[j])

            # Vibrate randomly
            positions[j] = positions[j] + vb * np.random.randn(dim)

            # Ensure positions stay within bounds
            positions[j] = np.clip(positions[j], lb, ub)

      # Get the best solution
    Best_pos = positions[0]
    Best_score = objf(Best_pos)
    final_checkpoint = torch.load('temp_SMA_best_model_min_val_loss.pth')
    torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'SMA_best_model.pth')

    return Best_pos, Best_score

def fitness_function_factory_SMA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion):
    def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion):
        #create intermediate fusion head for fused MLP models
        model, train_loaders, val_loaders, _ = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model_path = 'temp_SMA_best_model_min_val_loss.pth'
        #train and validate fused MLP models with fusion head
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']                                               #Validation results of intermediate_fusion_SMA
        return loss

    def fitness_func_SMA(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
        return solution_fitness
    return fitness_func_SMA

#------------------------------------------------- SMA optimization----------------------------------------------------------
def intermediate_fusion_SMA(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, SearchAgents_no, criterion):
    fitness_func_SMA = fitness_function_factory_SMA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    # Calculate number of fused MLP models + fusion head where to optimize the number of NN layers
    dim = sum(1 for data_type in dimension_dict.keys() for i in loaders_dict["train"][data_type]) + 1
    # Lower and upper bounds of number of layers
    lb, ub = 1, 10
    # return the best combination of NN layers and its loss
    best_solution, fitness = SMA(fitness_func_SMA, lb, ub, dim, SearchAgents_no, max_iter)
    os.remove('temp_SMA_best_model_min_val_loss.pth')
    model, _, _, test_loaders = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, np.around(best_solution))
    test_model = load_model(model, 'SMA_best_model.pth')
    test_loss = new_validate_intermediate(test_model, test_loaders, criterion, device)     #test model with the best combination of layers
    return np.around(best_solution).astype(int), test_loss
