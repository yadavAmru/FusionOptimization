import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from Dataset import attr_dim, img_dim, train_loader, val_loader
from Dataset import train_attr_loader, train_image_loader, val_attr_loader, val_image_loader
from early_fusion import early_fusion
from brute_force_search import brute_force_search
from intermediate_fusion_GA import interm_fusion_GA
from interm_fusion_GWO import interm_fusion_GWO
from interm_fusion_PSO import interm_fusion_PSO
from late_fusion import late_fusion

#-----------------------------------------------------Fusion results------------------------------------------------------------------
if __name__ == "__main__":
#Hyperparameters
    num_epochs = 3
    lr = 0.01
    criterion=nn.L1Loss()
    num_generations = 3 # The number of iterations in GA
    Max_iter = 3 # The number of iterations in GWO
    SearchAgents_no = 3 # Number of wolves seeking value in GWO
    num_particles = 2 # The number of particles in PSO
    max_iter = 5 # The number of iterations in PSO
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("------------------------------------------------------------Training--------------------------------------------------------------------------")
#Training fusion models
    print("-----------------------------------------------------Start of early fusion--------------------------------------------------------------------")
    early_fusion_loss = early_fusion(attr_dim, img_dim, train_loader, val_loader, device, lr, num_epochs, criterion)
    print("-------------------------------------Start of intermediate fusion with brute-force search-----------------------------------------------------")
    brute_force_loss, solution0, solution1 = brute_force_search(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, criterion)
    print("---------------------------------------------Start of intermediate fusion with GA-------------------------------------------------------------")
    solution_GA, interm_fusion_loss_GA = interm_fusion_GA(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, num_generations, criterion)
    print("---------------------------------------------Start of intermediate fusion with GWO------------------------------------------------------------")
    solution_GWO, interm_fusion_loss_GWO = interm_fusion_GWO(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, Max_iter, SearchAgents_no, criterion)
    print("---------------------------------------------Start of intermediate fusion with PSO------------------------------------------------------------")
    solution_PSO, interm_fusion_loss_PSO =  interm_fusion_PSO(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, max_iter, num_particles, criterion)
    print("-----------------------------------------------------Start of late fusion---------------------------------------------------------------------")
    late_fusion_loss = late_fusion(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, criterion)
    print("------------------------------------------------------------Results---------------------------------------------------------------------------")
    print("Early fusion loss: {:.5f}".format(early_fusion_loss))
    print("Brute force search loss: {:.5f} \t Best combination of NN layers: {}".format(brute_force_loss, [solution0, solution1]))
    print("Intermediate fusion loss (GA optimization): {:.5f} \t Best combination of NN layers: {}".format(1/(interm_fusion_loss_GA), solution_GA))
    print("Intermediate fusion loss (GWO optimization): {:.5f} \t Best combination of NN layers: {}".format(interm_fusion_loss_GWO, solution_GWO))
    print("Intermediate fusion loss (PSO optimization): {:.5f} \t Best combination of NN layers: {}".format(interm_fusion_loss_PSO, solution_PSO))
    print("Late fusion loss: {:.5f}".format(late_fusion_loss))      
    print("----------------------------------------------------------------------------------------------------------------------------------------------")
