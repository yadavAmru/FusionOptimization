import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from Dataset import img_dim, attr_dim, train_image_loader, train_attr_loader, val_image_loader, val_attr_loader
from late_fusion import late_fusion
from intermediate_fusion_GA import interm_fusion_GA

#-----------------------------------------------------Fusion results------------------------------------------------------------------
if __name__ == "__main__":
#Hyperparameters
    num_epochs = 1
    lr = 0.01
    criterion=nn.L1Loss()
    num_generations = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Training fusion models
    late_fusion_loss = late_fusion(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, criterion)
    solution_GA, interm_fusion_loss_GA = interm_fusion_GA(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, num_generations, criterion)
    print("-------------------------------------Results-----------------------------------------------------")
    print("Late fusion loss: {:.2f}".format(late_fusion_loss))
    print("Intermediate fusion loss (GA optimization): {:.2f} \t Solution: {}".format(interm_fusion_loss_GA, solution_GA))    
    print("-------------------------------------------------------------------------------------------------")