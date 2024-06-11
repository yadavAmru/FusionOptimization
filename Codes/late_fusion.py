import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from early_fusion import MLP, train

#----------------------------------------------------Late fusion---------------------------------------------------------------
def late_fusion(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr=0.01, num_epochs=1, criterion=nn.L1Loss()):
    mlp_num = MLP(input_dim=attr_dim, fusion="late")
    mlp_img = MLP(input_dim=img_dim, fusion="late")
    optimizer1 = optim.Adam(mlp_num.parameters(), lr=lr)
    optimizer2 = optim.Adam(mlp_img.parameters(), lr=lr)
    #Training
    mlp_num_path = 'best_num_late_fusion_model_min_val_loss.pth'
    mlp_img_path = 'best_img_late_fusion_model_min_val_loss.pth'
    num_dict_log = train(mlp_num, optimizer1, num_epochs, train_attr_loader, val_attr_loader, criterion, device, mlp_num_path)
    img_dict_log = train(mlp_img, optimizer2, num_epochs, train_image_loader, val_image_loader, criterion, device, mlp_img_path)
    #Results of late fusion
    checkpoint1 = torch.load('best_num_late_fusion_model_min_val_loss.pth')
    checkpoint2 = torch.load('best_img_late_fusion_model_min_val_loss.pth')
    loss = (checkpoint1['loss'] + checkpoint2['loss'])/2
    return loss
