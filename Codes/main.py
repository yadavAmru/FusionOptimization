import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import time
from Gas_Detection_Dataset import dimension_dict_gas, loaders_dict_gas
from early_fusion import early_fusion
from late_fusion import late_fusion
from intermediate_fusion_brute_force_search import intermediate_fusion_brute_force_search
from intermediate_fusion_GA import intermediate_fusion_GA
from intermediate_fusion_GWO import intermediate_fusion_GWO
from intermediate_fusion_PSO import intermediate_fusion_PSO
from intermediate_fusion_SMA import intermediate_fusion_SMA
from intermediate_fusion_SAA import intermediate_fusion_SAA
from intermediate_fusion_SHC import intermediate_fusion_SHC

#---------------------------------------------------------------Fusion results------------------------------------------------------------------------
if __name__ == "__main__":
    #Hyperparameters
    num_epochs = 20
    lr = 0.001
    ub = 15
    criterion=nn.L1Loss()
    num_generations = 5 # The number of iterations in GA
    Max_iter = 5 # The number of iterations in GWO
    max_iter = 5 # The number of iterations in PSO
    SearchAgents_no = 5 # Number of wolves seeking value in GWO
    num_particles = 5 # The number of particles in PSO
    sol_per_pop = 1 # The number of solutions per step in GA
    num_parents_mating = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("------------------------------------------------------------Training--------------------------------------------------------------------------")
    #Training fusion models
    print("-----------------------------------------------------Start of early fusion--------------------------------------------------------------------")
    start_time_early_fusion = time.time()
    early_fusion_loss, early_fusion_dict_log = early_fusion(dimension_dict_gas, loaders_dict_gas, device, lr, num_epochs, mode='classification', criterion=criterion)
    end_time_early_fusion = time.time()
    print("-----------------------------------------------------Start of late fusion---------------------------------------------------------------------")
    start_time_late_fusion = time.time()
    late_fusion_loss, late_fusion_dict_log = late_fusion(dimension_dict_gas, loaders_dict_gas, device, lr, num_epochs, mode='classification', criterion=criterion)
    end_time_late_fusion = time.time()
    print("-------------------------------------Start of intermediate fusion with brute-force search-----------------------------------------------------")
    start_time_brute_force_search = time.time()
    brute_force_solution, intermediate_brute_force_loss = intermediate_fusion_brute_force_search(dimension_dict_gas, loaders_dict_gas, device, ub, lr, num_epochs, mode='classification', criterion=criterion)
    end_time_brute_force_search = time.time()
    print("---------------------------------------------Start of intermediate fusion with GA-------------------------------------------------------------")
    start_time_GA = time.time()
    solution_GA, intermediate_fusion_loss_GA = intermediate_fusion_GA(dimension_dict_gas, loaders_dict_gas, device, ub, lr, num_epochs, num_generations, sol_per_pop, num_parents_mating, mode='classification', criterion=criterion)
    end_time_GA = time.time()
    print("---------------------------------------------Start of intermediate fusion with GWO------------------------------------------------------------")
    start_time_GWO = time.time()
    solution_GWO, intermediate_fusion_loss_GWO = intermediate_fusion_GWO(dimension_dict_gas, loaders_dict_gas, device, ub, lr, num_epochs, Max_iter, SearchAgents_no, mode='classification', criterion=criterion)
    end_time_GWO = time.time()
    print("---------------------------------------------Start of intermediate fusion with PSO------------------------------------------------------------")
    start_time_PSO = time.time()
    solution_PSO, intermediate_fusion_loss_PSO = intermediate_fusion_PSO(dimension_dict_gas, loaders_dict_gas, device, ub, lr, num_epochs, max_iter, num_particles, mode='classification', criterion=criterion)
    end_time_PSO = time.time()
    print("---------------------------------------------Start of intermediate fusion with SMA------------------------------------------------------------")
    start_time_SMA = time.time()
    solution_SMA, intermediate_fusion_loss_SMA = intermediate_fusion_SMA(dimension_dict_gas, loaders_dict_gas, device, ub, lr, num_epochs, max_iter, SearchAgents_no, mode='classification', criterion=criterion)
    end_time_SMA = time.time()
    print("---------------------------------------------Start of intermediate fusion with SAA------------------------------------------------------------")
    start_time_SAA = time.time()
    solution_SAA, intermediate_fusion_loss_SAA = intermediate_fusion_SAA(dimension_dict_gas, loaders_dict_gas, device, ub, lr, num_epochs, max_iter, mode='classification', criterion=criterion)
    end_time_SAA = time.time()
    print("---------------------------------------------Start of intermediate fusion with SHC------------------------------------------------------------")
    start_time_SHC = time.time()
    solution_SHC, intermediate_fusion_loss_SHC = intermediate_fusion_SHC(dimension_dict_gas, loaders_dict_gas, device, ub, lr, num_epochs, max_iter, mode='classification', criterion=criterion)
    end_time_SHC = time.time()
    print("------------------------------------------------------------Test results---------------------------------------------------------------------------")
    print("Early fusion test loss: {:.5f} \t Time: {:.3f} seconds".format(early_fusion_loss, end_time_early_fusion - start_time_early_fusion))
    print("Late fusion test loss: {:.5f} \t Time: {:.3f} seconds".format(late_fusion_loss, end_time_late_fusion - start_time_late_fusion))
    print("Brute force search test loss: {:.5f} \t Best combination of NN layers: {} \t Time: {:.3f} seconds".format(intermediate_brute_force_loss, brute_force_solution, end_time_brute_force_search - start_time_brute_force_search))
    print("Intermediate fusion test loss (GA optimization): {:.5f} \t Best combination of NN layers: {} \t Time: {:.3f} seconds".format(intermediate_fusion_loss_GA, solution_GA, end_time_GA - start_time_GA))
    print("Intermediate fusion test loss (GWO optimization): {:.5f} \t Best combination of NN layers: {} \t Time: {:.3f} seconds".format(intermediate_fusion_loss_GWO, solution_GWO, end_time_GWO - start_time_GWO))
    print("Intermediate fusion test loss (PSO optimization): {:.5f} \t Best combination of NN layers: {} \t Time: {:.3f} seconds".format(intermediate_fusion_loss_PSO, solution_PSO, end_time_PSO - start_time_PSO))
    print("Intermediate fusion test loss (SMA optimization): {:.5f} \t Best combination of NN layers: {} \t Time: {:.3f} seconds".format(intermediate_fusion_loss_SMA, solution_SMA, end_time_SMA - start_time_SMA))
    print("Intermediate fusion test loss (SAA optimization): {:.5f} \t Best combination of NN layers: {} \t Time: {:.3f} seconds".format(intermediate_fusion_loss_SAA, solution_SAA, end_time_SAA - start_time_SAA))
    print("Intermediate fusion test loss (SHC optimization): {:.5f} \t Best combination of NN layers: {} \t Time: {:.3f} seconds".format(intermediate_fusion_loss_SHC, solution_SHC, end_time_SHC - start_time_SHC))
    print("----------------------------------------------------------------------------------------------------------------------------------------------")
