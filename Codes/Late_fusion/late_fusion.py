import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from early_fusion import MLP, train, validate, load_model

#----------------------------------------------------Late fusion---------------------------------------------------------------
def late_fusion(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion):
    loss_array = []
    dict_log = {}
    for data_type in dimension_dict.keys():
        for i, (train_loader, val_loader, test_loader) in enumerate(zip(loaders_dict["train"][data_type], loaders_dict["val"][data_type], loaders_dict["test"][data_type])):
            model = MLP(input_dim=dimension_dict[data_type], fusion="late", mode=mode)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model_path = 'best_late_fusion_model_' + data_type + '_' + str(i) + '.pth'
            log = train(model, optimizer, num_epochs, train_loader, val_loader, criterion, device, model_path)   #train model with validation
            dict_log[data_type] = log
            # checkpoint = torch.load(model_path)
            # loss_array.append(checkpoint['loss'])                               #Collect all validation losses for each modality
            test_model = load_model(model, model_path)
            test_loss = validate(test_model, test_loader, criterion, device)
            loss_array.append(test_loss)                                        #Collect all test losses for each modality
    test_loss = sum(loss_array) / len(loss_array)                               #Find mean value fo all validation results to get final loss for late fusion
    return test_loss, dict_log
