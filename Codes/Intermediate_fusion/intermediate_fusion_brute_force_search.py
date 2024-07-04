import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import binary_repr
from early_fusion import MLP

#-------------------------------------------Definition of functions---------------------------------------------------------
class Fusion_Head(nn.Module):
    def __init__(self, model_list, n_layers = 5, output_dim=1):
        super(Fusion_Head, self).__init__()
        self.model_list = model_list
        input_dim = 0
        for i in range(len(self.model_list)):
            input_dim += list(self.model_list[i].children())[0][-2].out_features

        if n_layers == None:                              # if we do not receive the number of layers -> halving (output nodes = input nodes / 2) of the nodes for each neural network layer starts
            layer_list = nn.ModuleList()
            in_nodes = input_dim
            if (input_dim // 2) % 2 != 0:                 # if we have odd number of input nodes, we turn it to even number and start halving
                out_nodes = input_dim // 2 + 1
                layer_list.append(nn.Linear(in_nodes, out_nodes))
                layer_list.append(nn.ReLU())
                in_nodes = out_nodes
            else:                                         # if we have even number of input nodes, we start halving without any changes
                out_nodes = input_dim // 2
            while out_nodes % 2 == 0 and (in_nodes // 2) != 1 and out_nodes != 0:
                out_nodes = in_nodes//2
                layer_list.append(nn.Linear(in_nodes, out_nodes))
                layer_list.append(nn.ReLU())
                in_nodes = out_nodes
            self.layers = nn.Sequential(*layer_list)
            out_nodes = out_nodes
        else:                         # if we receive the number of layers -> we linearly decrease number of nodes for each neural network layer
            layer_list = nn.ModuleList()
            nodes = np.linspace(input_dim, 1, num=n_layers + 1, dtype=int)
            for idx in range(len(nodes) - 2):
                nodes[idx], nodes[idx + 1]
                layer_list.append(nn.Linear(nodes[idx], nodes[idx + 1]))
                layer_list.append(nn.ReLU())
            self.layers = nn.Sequential(*layer_list)
            out_nodes = nodes[-2]
        self.output_fc = nn.Linear(out_nodes, output_dim)

    def forward(self, x_list):
        models = []
        for i in range(len(self.model_list)):
            model = self.model_list[i](x_list[i].float())
            models.append(model)
        x = torch.cat(models, dim=1)
        x = self.layers(x)
        output = self.output_fc(x)
        return output

def new_train_one_epoch_intermediate(model, optimizer, train_loaders, criterion, device):
    model.train()
    loss_step = []
    num_loaders = len(train_loaders)
    for data in zip(*[train_loaders[i] for i in range(num_loaders)]):
        #Collect inputs and labels from all provided data loaders
        inputs, labels =  [data[i][0][0].to(device) for i in range(num_loaders)], [data[i][1][0].to(device) for i in range(num_loaders)]
        outputs = model(inputs)                                         #Forward propagation
        loss = criterion(outputs, labels[0])                            #Compute loss function
        optimizer.zero_grad()
        loss.backward()                                                 #Backward propagation
        optimizer.step()                                                #Update parameters
        loss_step.append(loss.item())
    loss_curr_epoch = np.mean(loss_step)
    return loss_curr_epoch

def new_validate_intermediate(model, val_loaders, criterion, device):
    model.eval()
    loss_step = []
    num_loaders = len(val_loaders)
    with torch.no_grad():
        for data in zip(*[val_loaders[i] for i in range(num_loaders)]):
            #Collect inputs and labels from all provided data loaders
            inputs, labels =  [data[i][0][0].to(device) for i in range(num_loaders)], [data[i][1][0].to(device) for i in range(num_loaders)]
            outputs = model(inputs)                                             #Forward propagation
            val_loss = criterion(outputs, labels[0])                            #Compute loss function
            loss_step.append(val_loss.item())
    val_loss_epoch = torch.tensor(loss_step).mean().numpy()
    return val_loss_epoch

def new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, path):
    dict_log = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    model = model.to(device)
    for epoch in range(num_epochs):
        loss_curr_epoch = new_train_one_epoch_intermediate(model, optimizer, train_loaders, criterion, device)      #train for one epoch
        val_loss = new_validate_intermediate(model, val_loaders, criterion, device)               #validate in the same epoch
        # Print epoch results to screen
        print(f'Ep {epoch + 1}/{num_epochs}: Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
        dict_log["train_loss"].append(loss_curr_epoch)
        dict_log["val_loss"].append(val_loss)
        # Use this code to save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f'{path}', epoch, optimizer, val_loss)
    return dict_log

def save_model(model, path, epoch, optimizer, val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        }, path)

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



def get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution):
    train_loaders, val_loaders, test_loaders = [], [], []
    models = []
    index = 0
    for data_type in dimension_dict.keys():        #we go through each dataloader for each modality and create separate MLP model for them
        train_loaders_per_type, val_loaders_per_type, test_loaders_per_type = [], [], []
        for i, (train_loader, val_loader, test_loader) in enumerate(zip(loaders_dict["train"][data_type], loaders_dict["val"][data_type], loaders_dict["test"][data_type])):
            train_loaders_per_type.append(train_loader)
            val_loaders_per_type.append(val_loader)
            test_loaders_per_type.append(test_loader)
            input_dim = dimension_dict[data_type]
            model = MLP(input_dim=input_dim, n_layers=round(solution[index]), fusion="intermediate")
            models.append(model)
            index += 1
        train_loaders.extend(train_loaders_per_type)                            #collect all train data loaders
        val_loaders.extend(val_loaders_per_type)                                #collect all validation data loaders
        test_loaders.extend(test_loaders_per_type)                              #collect all test data loaders
    model = Fusion_Head(models, n_layers=round(solution[index]))                #create intermediate fusion head for fused MLP models
    return model, train_loaders, val_loaders, test_loaders

#-------------------------------------------Main part---------------------------------------------------------
def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion):
    #create intermediate fusion head for fused MLP models
    model, train_loaders, val_loaders, _ = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #Training
    model_path = 'temp_Brute_force_search_best_model_min_val_loss.pth'
    print("Combination of layers: {}".format(solution))
    #train and validate fused MLP models with fusion head
    dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
    checkpoint = torch.load(model_path)
    loss = checkpoint['loss']                                           #Validation results of brute-force search
    return loss

def intermediate_fusion_brute_force_search(dimension_dict, loaders_dict, device, lr=0.01, num_epochs=1, criterion=nn.L1Loss()):
    # Upper bound of number of layers
    ub = 10
    # Calculate number of fused MLP models + fusion head where to optimize the number of NN layers
    num_solutions = sum(1 for data_type in dimension_dict.keys() for i in loaders_dict["train"][data_type]) + 1
    combinations = 2 ** num_solutions
    orig_sol = np.ones(num_solutions, dtype=int)
    best_loss = float("inf")
    best_solution = orig_sol
    best_combo = {'loss': best_loss, 'solution': best_solution}
    # calculate combination of layers by incrementing number of layers in each MLP model
    # one by one till upper boundary: [1,1,1] -> [1,1,2] -> [1,2,2] -> [2,2,2] -> [1,2,3] -> ...
    for i in range(ub - 1):
        for combination in range(combinations):
            solution_arr = binary_repr(combination, width = len(binary_repr(combinations)) - 1)
            solution_arr = list(solution_arr)
            solution = orig_sol + [int(sol) for sol in solution_arr]
            loss = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
            if loss < best_loss:
                best_loss = loss
                best_combo['loss'] = best_loss
                best_combo['solution'] = solution
                final_checkpoint = torch.load('temp_Brute_force_search_best_model_min_val_loss.pth')
                torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'Brute_force_search_best_model.pth')
        orig_sol = orig_sol + np.ones(num_solutions, dtype=int)
    # return the best combination of NN layers and its loss
    os.remove('temp_Brute_force_search_best_model_min_val_loss.pth')
    model, _, _, test_loaders = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, best_combo['solution'])
    test_model = load_model(model, 'Brute_force_search_best_model.pth')
    test_loss = new_validate_intermediate(test_model, test_loaders, criterion, device)
    return best_combo['solution'], test_loss
