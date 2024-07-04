import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import numpy as np
from intermediate_fusion_brute_force_search import new_train_intermediate, new_validate_intermediate
from intermediate_fusion_brute_force_search import get_fusion_model_and_dataloaders, load_model

#-------------------------------------------Definition of functions---------------------------------------------------------
def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -float("inf")
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 2] = offspring_crossover[idx, 2] + random_value
    return offspring_crossover

def GA(objf, lb, ub, num_weights, num_generations, sol_per_pop, num_parents_mating):
    # Defining the population size.
    pop_size = (sol_per_pop, num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    #Creating the initial population.
    new_population = numpy.random.uniform(low=lb, high=ub, size=pop_size)

    num_generations = num_generations
    for generation in range(num_generations):
        # Measing the fitness of each chromosome in the population.
        fitness = []
        for sol in new_population:
            sol = [round(x) for x in sol]
            fitness.append(objf(sol))

        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(new_population, fitness,
                                          num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents,
                                          offspring_size=(pop_size[0]-parents.shape[0], num_weights))

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = mutation(offspring_crossover)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        # The best result in the current iteration.
        # print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.
    fitness = []
    for sol in new_population:
        sol = [round(x) for x in sol]
        fitness.append(objf(sol))

    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = numpy.where(fitness == numpy.max(fitness))
    best_solution = new_population[best_match_idx, :][0][0]
    best_solution = [round(x) for x in best_solution]
    solution_fitness = fitness[best_match_idx[0][0]]

    best_model_save = objf(best_solution)
    final_checkpoint = torch.load('temp_GA_best_model_min_val_loss.pth')
    torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'GA_best_model.pth')

    return best_solution, solution_fitness


def fitness_function_factory_GA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion):
    def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion):
        #create intermediate fusion head for fused MLP models
        model, train_loaders, val_loaders, _ = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model_path = 'temp_GA_best_model_min_val_loss.pth'
        #train and validate fused MLP models with fusion head
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']                                               #Validation results of intermediate_fusion_GA
        return loss

    def fitness_func_GA(solution):
        loss = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
        solution_fitness = 1 / (loss + 1e-6)
        return solution_fitness
    return fitness_func_GA

#------------------------------------------------- GA optimization----------------------------------------------------------
def intermediate_fusion_GA(dimension_dict, loaders_dict, device, lr, num_epochs, num_generations, sol_per_pop, num_parents_mating, criterion):
    fitness_func_GA = fitness_function_factory_GA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    # Calculate number of fused MLP models + fusion head where to optimize the number of NN layers
    num_weights = sum(1 for data_type in dimension_dict.keys() for i in loaders_dict["train"][data_type]) + 1
    # Lower and upper bounds of number of layers
    lb, ub = 1, 10
    # return the best combination of NN layers and its loss
    best_solution, solution_fitness = GA(fitness_func_GA, lb, ub, num_weights, num_generations, sol_per_pop, num_parents_mating)
    os.remove('temp_GA_best_model_min_val_loss.pth')
    model, _, _, test_loaders = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, np.around(best_solution))
    test_model = load_model(model, 'GA_best_model.pth')
    test_loss = new_validate_intermediate(test_model, test_loaders, criterion, device)      #test model with the best combination of layers
    return np.around(best_solution).astype(int), test_loss
