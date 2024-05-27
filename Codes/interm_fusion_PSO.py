import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy as np
from intermediate_fusion_GA import MLP_intermediate, MyEnsemble, train_intermediate
from late_fusion import save_model, load_model

#-------------------------------------------Definition of functions---------------------------------------------------------
# Define the PSO algorithm
def PSO(cost_func, dim=2, num_particles=3, max_iter=5, w=0.5, c1=1, c2=2, lb=1, ub=5):
    # Initialize particles and velocities
    particles = np.random.uniform(lb, ub, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities

        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

        print([' The number of iterations is ' + str(i) + '  Iterative results of ' + str(swarm_best_fitness) + '; Solution: ' + str(np.around(swarm_best_position))]);  #  Results of each iteration

    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness


def fitness_func_PSO(solution):
    new_mlp_img = MLP_intermediate(input_dim=img_dim, n_layers = round(solution[0]))
    new_mlp_num = MLP_intermediate(input_dim=attr_dim, n_layers = round(solution[1]))
    model = MyEnsemble(new_mlp_img, new_mlp_num)
    optimizer_intermediate = optim.Adam(model.parameters(), lr=lr)
    path = 'PSO_best_model_min_val_loss.pth'
    dict_log = train_intermediate(model, optimizer_intermediate, num_epochs, train_image_loader, train_attr_loader, val_image_loader, val_attr_loader, criterion, device, path)
    checkpoint = torch.load('PSO_best_model_min_val_loss.pth')
    loss = checkpoint['loss']
    return loss

#------------------------------------------------- PSO optimization----------------------------------------------------------
def interm_fusion_PSO(attr_dim_PSO, img_dim_PSO, train_attr_loader_PSO, train_image_loader_PSO, val_attr_loader_PSO, val_image_loader_PSO, device_PSO, lr_PSO, num_epochs_PSO, max_iter, num_particles, criterion_PSO):
    global attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, criterion
    attr_dim, img_dim = attr_dim_PSO, img_dim_PSO
    train_attr_loader, train_image_loader = train_attr_loader_PSO, train_image_loader_PSO
    val_attr_loader, val_image_loader = val_attr_loader_PSO, val_image_loader_PSO
    device, lr, num_epochs, criterion = device_PSO, lr_PSO, num_epochs_PSO, criterion_PSO
    lb, ub = 1, 5 # Lower and upper bounds
    dim = 2 # Search range of wolf
    w, c1, c2 = 0.5, 1, 2 #inertia weight, personal acceleration factor, social acceleration factor
    solution, fitness = PSO(fitness_func_PSO, dim=dim, num_particles=num_particles, max_iter=max_iter, w=w, c1=c1, c2=c2, lb=1, ub=5)
    return np.around(solution), fitness