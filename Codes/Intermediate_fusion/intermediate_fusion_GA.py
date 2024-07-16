#pip install pygad
import pygad
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from intermediate_fusion_brute_force_search import new_train_intermediate, new_validate_intermediate
from intermediate_fusion_brute_force_search import get_fusion_model_and_dataloaders, load_model

#-------------------------------------------Definition of functions---------------------------------------------------------
def fitness_function_factory_GA(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion):
    def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion):
        #create intermediate fusion head for fused MLP models
        model, train_loaders, val_loaders, _ = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution, mode, device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        model_path = 'temp_GA_best_model_min_val_loss.pth'
        #train and validate fused MLP models with fusion head
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']                                               #Validation results of intermediate_fusion_GA
        return loss

    def fitness_func_GA(ga_instance, solution, solution_idx):
        loss = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion)
        solution_fitness = 1 / (loss + 1e-6)
        return solution_fitness
    return fitness_func_GA

def callback_generation(ga_instance):
    print("Generation: {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Best solution = {fitness}".format(fitness=ga_instance.best_solution()[0]))

#------------------------------------------------- GA optimization----------------------------------------------------------
def intermediate_fusion_GA(dimension_dict, loaders_dict, device, lr, num_epochs, num_generations, sol_per_pop, num_parents_mating, mode, criterion):
    fitness_func_GA = fitness_function_factory_GA(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion)
    # Calculate number of fused MLP models + fusion head where to optimize the number of NN layers
    num_genes = sum(1 for data_type in dimension_dict.keys() for i in loaders_dict["train"][data_type])
    # Lower and upper bounds of number of layers
    gene_space = [[1, int(np.log2(up_b))] for up_b in dimension_dict.values()]
    # return the best combination of NN layers and its loss
    ga_instance = pygad.GA(num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitness_func_GA,
                    on_generation=callback_generation,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes, # Two solutions (one for each model)
                    gene_type=int,
                    gene_space=gene_space,  #  Range for number of layers for each model
                    )
    ga_instance.run()
    best_solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_model_save = fitness_func_GA(ga_instance, best_solution, solution_idx)
    final_checkpoint = torch.load('temp_GA_best_model_min_val_loss.pth')
    torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'GA_best_model.pth')
    os.remove('temp_GA_best_model_min_val_loss.pth')
    model, _, _, test_loaders = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, np.around(best_solution), mode, device)
    test_model = load_model(model, 'GA_best_model.pth')
    test_loss = new_validate_intermediate(test_model, test_loaders, criterion, device)      #test model with the best combination of layers
    return np.around(best_solution).astype(int), test_loss
