import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy as np
import copy    # array-copying convenience
import sys     # max float
from intermediate_fusion_brute_force_search import new_train_intermediate, new_validate_intermediate
from intermediate_fusion_brute_force_search import get_fusion_model_and_dataloaders, load_model

#-------------------------------------------Definition of functions---------------------------------------------------------
# Define the PSO algorithm
#particle class
class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        # initialize position of the particle with 0.0 value
        self.position = [0.0 for i in range(dim)]
        # initialize velocity of the particle with 0.0 value
        self.velocity = [0.0 for i in range(dim)]
        # initialize best particle position of the particle with 0.0 value
        self.best_part_pos = [0.0 for i in range(dim)]
        # loop dim times to calculate random position and velocity
        # range of position and velocity is [minx, max]
        for i in range(dim):
            self.position[i] = ((maxx - minx) *
                self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx) *
                self.rnd.random() + minx)

        # compute fitness of particle
        self.fitness = fitness(self.position) # curr fitness
        # initialize best position and fitness of this particle
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness # best fitness


# particle swarm optimization function
def PSO(fitness, max_iter, n, dim, minx, maxx):
    # hyper parameters
    w = 0.729    # inertia
    c1 = 1.49445 # cognitive (particle)
    c2 = 1.49445 # social (swarm)

    rnd = random.Random(0)

    # create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]

    # compute the value of best_position and best_fitness in swarm
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max # swarm best

    # computer best particle of swarm and it's fitness
    for i in range(n): # check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)
            # final_checkpoint = torch.load('temp_PSO_best_model_min_val_loss.pth')
            # torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'PSO_best_model.pth')

    # main loop of pso
    Iter = 0
    while Iter < max_iter:
        # after every 10 iterations
        # print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)

        for i in range(n): # process each particle
            # compute new velocity of curr particle
            for k in range(dim):
                r1 = rnd.random()    # randomizations
                r2 = rnd.random()

                swarm[i].velocity[k] = (
                                        (w * swarm[i].velocity[k]) +
                                        (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) +
                                        (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k]))
                                      )

                # if velocity[k] is not in [minx, max]
                # then clip it
                if swarm[i].velocity[k] < minx:
                    swarm[i].velocity[k] = minx
                elif swarm[i].velocity[k] > maxx:
                    swarm[i].velocity[k] = maxx

            # compute new position using new velocity
            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                if swarm[i].position[k] < minx:
                    swarm[i].position[k] = minx
                elif swarm[i].position[k] > maxx:
                    swarm[i].position[k] = maxx
                # print("Swarm positions: ", swarm[i].position[k])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # compute fitness of new position
            swarm[i].fitness = fitness(swarm[i].position)

            # is new position a new best for the particle?
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)
                # final_checkpoint = torch.load('temp_PSO_best_model_min_val_loss.pth')
                # torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'PSO_best_model.pth')

        # for-each particle
        Iter += 1
    #end_while

    best_model_save = fitness(best_swarm_pos)
    final_checkpoint = torch.load('temp_PSO_best_model_min_val_loss.pth')
    torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'PSO_best_model.pth')
    return best_swarm_pos, best_swarm_fitnessVal
  # end pso


def fitness_function_factory_PSO(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion):
    def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion):
        #create intermediate fusion head for fused MLP models
        model, train_loaders, val_loaders, _ = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution, mode, device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        model_path = 'temp_PSO_best_model_min_val_loss.pth'
        #train and validate fused MLP models with fusion head
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']                                               #Validation results of intermediate_fusion_PSO
        return loss

    def fitness_func_PSO(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion)
        return solution_fitness
    return fitness_func_PSO

#------------------------------------------------- PSO optimization----------------------------------------------------------
def intermediate_fusion_PSO(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, num_particles, mode, criterion):
    fitness_func_PSO = fitness_function_factory_PSO(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion)
    # Calculate number of fused MLP models + fusion head where to optimize the number of NN layers
    dim = sum(1 for data_type in dimension_dict.keys() for i in loaders_dict["train"][data_type])
    # Lower and upper bounds of number of layers
    lb, ub = 1, int(np.mean([int(np.log2(up_b)) for up_b in dimension_dict.values()]))
    # return the best combination of NN layers and its loss
    best_solution, fitness = PSO(fitness_func_PSO, max_iter, num_particles, dim, lb, ub)
    os.remove('temp_PSO_best_model_min_val_loss.pth')
    model, _, _, test_loaders = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, np.around(best_solution), mode, device)
    test_model = load_model(model, 'PSO_best_model.pth')
    test_loss = new_validate_intermediate(test_model, test_loaders, criterion, device)      #test model with the best combination of layers
    return np.around(best_solution).astype(int), test_loss
