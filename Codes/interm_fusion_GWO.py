import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy
from intermediate_fusion_GA import MLP_intermediate, MyEnsemble, train_intermediate
from late_fusion import save_model, load_model

#-------------------------------------------Definition of functions---------------------------------------------------------
def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    #=== initialization  alpha, beta, and delta_pos=======
    Alpha_pos = numpy.zeros(dim)  #  Location . formation 30 A list of
    Alpha_score = float("inf")  #  This means “ Plus or minus infinity ”, All the numbers are equal to  +inf  Small ; It's just infinite ：float("inf");  Negative infinity ：float("-inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")  # float()  Function to convert integers and strings to floating-point numbers .

    #====list List the type =============
    if not isinstance(lb, list):  #  effect ： To determine whether an object is a known type . Its first parameter （object） As object , The second parameter （type） For type name , If the type of the object is the same as that of parameter 2, return True
        lb = [lb] * dim  #  Generate [100,100,.....100]30 individual
    if not isinstance(ub, list):
        ub = [ub] * dim

    #======== Initialize the location of all wolves ===================
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  #  formation 5*30 Number [-100,100) within
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[
            i]  #  formation [5 individual 0-1 Number of numbers ]*100-（-100）-100
    Convergence_curve = numpy.zeros(Max_iter)

    #======== Iterative optimization =====================
    for l in range(0, Max_iter):  #  iteration 1000
        for i in range(0, SearchAgents_no):  # 5
            #==== Returns the search agent that exceeds the boundary of the search space ====
            for j in range(dim):  # 30
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])  # clip This function limits the elements in the array to a_min(-100), a_max(100) Between , Greater than a_max That makes it equal to  a_max, Less than a_min, That makes it equal to a_min.

            #=== Calculate the objective function of each search agent ==========
            fitness = objf(Positions[i, :])  #  Bring a row of data into function calculation
            # print(" After calculation ：",fitness)

            #==== to update  Alpha, Beta, and Delta================
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        #=========== In the above cycle ,Alpha、Beta、Delta===========
        a = 2 - l * ((2) / Max_iter);  # a from 2 Linear reduction to 0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1] Mainly generate one 0-1 Is a random floating point number .
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  # Equation (3.3)
                C1 = 2 * r2;  # Equation (3.4)
                # D_alpha Indicates the candidate wolf and Alpha The distance of the wolf
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[
                    i, j]);  # abs()  Function returns the absolute value of a number .Alpha_pos[j] Express Alpha Location ,Positions[i,j]) Location of candidate gray wolf
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1 Express basis alpha The position vector of the next generation gray wolf

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;

                Positions[i, j] = (X1 + X2 + X3) / 3  #  The location of the candidate wolf is updated according to Alpha、Beta、Delta The address of the next generation of gray wolf .

        Convergence_curve[l] = Alpha_score;

        if (l % 1 == 0):
            print([' The number of iterations is ' + str(l) + '  Iterative results of ' + str(Alpha_score)]);  #  Results of each iteration
    print("The best optimal value of the objective funciton found by GWO is: {:.5f} \t Best solution: {}".format(Alpha_score, numpy.around(Alpha_pos)))
    return Alpha_pos, Alpha_score

def fitness_func_GWO(solution):
    new_mlp_img = MLP_intermediate(input_dim=img_dim, n_layers = round(solution[0]))
    new_mlp_num = MLP_intermediate(input_dim=attr_dim, n_layers = round(solution[1]))
    model = MyEnsemble(new_mlp_img, new_mlp_num)
    optimizer_intermediate = optim.Adam(model.parameters(), lr=lr)
    path = 'GWO_best_model_min_val_loss.pth'
    dict_log = train_intermediate(model, optimizer_intermediate, num_epochs, train_image_loader, train_attr_loader, val_image_loader, val_attr_loader, criterion, device, path)
    checkpoint = torch.load('GWO_best_model_min_val_loss.pth')
    loss = checkpoint['loss']
    return loss

#------------------------------------------------- GWO optimization----------------------------------------------------------
def interm_fusion_GWO(attr_dim_GWO, img_dim_GWO, train_attr_loader_GWO, train_image_loader_GWO, val_attr_loader_GWO, val_image_loader_GWO, device_GWO, lr_GWO, num_epochs_GWO, Max_iter, SearchAgents_no, criterion_GWO):
    global attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, criterion
    attr_dim, img_dim = attr_dim_GWO, img_dim_GWO
    train_attr_loader, train_image_loader = train_attr_loader_GWO, train_image_loader_GWO
    val_attr_loader, val_image_loader = val_attr_loader_GWO, val_image_loader_GWO
    device, lr, num_epochs, criterion = device_GWO, lr_GWO, num_epochs_GWO, criterion_GWO
    lb, ub = 1, 5 # Lower and upper bounds
    dim = 2 # Search range of wolf
    solution, solution_fitness = GWO(fitness_func_GWO, lb, ub, dim, SearchAgents_no, Max_iter)
    return numpy.around(solution), solution_fitness