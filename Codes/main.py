import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from Dataset import img_dim, attr_dim, train_image_loader, train_attr_loader, val_image_loader, val_attr_loader
from late_fusion import late_fusion
from intermediate_fusion_GA import interm_fusion_GA
from brute_force_search import brute_force_search

#-----------------------------------------------------Fusion results------------------------------------------------------------------
if __name__ == "__main__":
#Hyperparameters
    num_epochs = 1
    lr = 0.01
    criterion=nn.L1Loss()
    num_generations = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Training fusion models
    brute_force_loss, solution0, solution1 = brute_force_search(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, criterion)
    solution_GA, interm_fusion_loss_GA = interm_fusion_GA(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, num_generations, criterion)
    late_fusion_loss = late_fusion(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, criterion)
    print("-------------------------------------Results-----------------------------------------------------")
    print("Brute force search loss: {:.4f} \t Best combination of NN layers: {}".format(brute_force_loss, [solution0, solution1]))
    print("Intermediate fusion loss (GA optimization): {:.4f} \t Best combination of NN layers: {}".format(1/(interm_fusion_loss_GA), solution_GA))    
    print("Late fusion loss: {:.4f}".format(late_fusion_loss))  
    print("-------------------------------------------------------------------------------------------------")
