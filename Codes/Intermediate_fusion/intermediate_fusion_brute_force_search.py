import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from early_fusion import MLP_early_late_fusion

#-------------------------------------------Definition of functions---------------------------------------------------------
class MLP_intermediate_fusion(nn.Module):
    def __init__(self, input_dim, orig_model, n_layers=None):
        super(MLP_intermediate_fusion, self).__init__()
        self.input_dim = input_dim
        layer_list = nn.ModuleList()
        if n_layers == 0:
            raise ValueError("Cannot have 0 layers!!!")
        elif n_layers != None:
            self.layers = orig_model.layers[:n_layers*2]

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.layers(x.float())
        return x
    
class Fusion_Head(nn.Module):
    def __init__(self, model_list, mode, n_layers = 5, output_dim=1):
        super(Fusion_Head, self).__init__()
        self.model_list = model_list
        self.mode = mode
        input_dim = 0
        layer_list = nn.ModuleList()
        for i in range(len(self.model_list)):
            input_dim += list(self.model_list[i].children())[0][-2].out_features

        if n_layers == None:              # if we do not receive the number of layers -> halving (output nodes = input nodes / 2) of the nodes for each neural network layer starts
            in_nodes = input_dim
            if input_dim == 0:                                  # we cannot have dimension = 0
                raise ValueError("Input dimension cannot be zero!!!")
            elif input_dim == 1 or input_dim == 2 or input_dim == 3:   # if input dimension = 1 or 2 or 3 -> we create just 1 linear layer with output dimension = 2
                out_nodes = 2
                layer_list.append(nn.Linear(in_nodes, out_nodes))
                layer_list.append(nn.ReLU())
            while (in_nodes // 2) != 0 and (in_nodes // 2) != 1 and in_nodes != 1:               # we create layers with output nodes twice less than input nodes
                out_nodes = in_nodes//2
                layer_list.append(nn.Linear(in_nodes, out_nodes))
                layer_list.append(nn.ReLU())
                in_nodes = out_nodes
            self.layers = nn.Sequential(*layer_list)
            out_nodes = out_nodes
        else:                         # if we receive the number of layers -> we linearly decrease number of nodes for each neural network layer
            nodes = np.linspace(input_dim, 1, num=n_layers + 1, dtype=int)
            for idx in range(len(nodes) - 2):
                nodes[idx], nodes[idx + 1]
                layer_list.append(nn.Linear(nodes[idx], nodes[idx + 1]))
                layer_list.append(nn.ReLU())
            self.layers = nn.Sequential(*layer_list)
            out_nodes = nodes[-2]
        self.output_fc = nn.Linear(out_nodes, output_dim)
        self.sigmoid_cls = nn.Sigmoid()

    def forward(self, x_list):
        models = []
        for i in range(len(self.model_list)):
            model = self.model_list[i](x_list[i].float())
            models.append(model)
        x = torch.cat(models, dim=1)
        x = self.layers(x)
        output = self.output_fc(x)
        if self.mode == "classification":       # for classification we need to add sigmoid activation function to the output of linear layer
            sigmoid_output = self.sigmoid_cls(output)
            return sigmoid_output
        elif self.mode == "regression":         # for regression we just return the output of linear layer
            return output
        else:
            raise ValueError("Incorrect model type selected!!!")
        return output

def new_train_one_epoch_intermediate(model, optimizer, train_loaders, criterion, device):
    model.train()
    loss_step = []
    num_loaders = len(train_loaders)
    for data in zip(*[train_loaders[i] for i in range(num_loaders)]):
        #Collect inputs and labels from all provided data loaders
        inputs, labels =  [data[i][0][0].to(device) for i in range(num_loaders)], [data[i][1][0].to(device) for i in range(num_loaders)]
        outputs = model(inputs)                                                       #Forward propagation
        loss = criterion(outputs.squeeze(1), labels[0].float().unsqueeze(0))          #Compute loss function
        optimizer.zero_grad()
        loss.backward()                                                               #Backward propagation
        optimizer.step()                                                              #Update parameters
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
            val_loss = criterion(outputs.squeeze(1), labels[0].float().unsqueeze(0))                    #Compute loss function
            loss_step.append(val_loss.item())
    val_loss_epoch = torch.tensor(loss_step).mean().numpy()
    return val_loss_epoch

def new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, path):
    dict_log = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    # model = model.to(device)
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



def get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution, mode, device):
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
            orig_model = MLP_early_late_fusion(input_dim=dimension_dict[data_type], mode=mode)
            # model_path = 'best_late_fusion_model_' + data_type + '_' + str(i) + '.pth'
            # orig_model = load_model(orig_model, model_path)
            # for param in orig_model.parameters():
            #     param.requires_grad = False
            model = MLP_intermediate_fusion(input_dim=dimension_dict[data_type], orig_model=orig_model, n_layers=round(solution[index]))
            model = model.to(device)
            models.append(model)
            index += 1
        train_loaders.extend(train_loaders_per_type)                            #collect all train data loaders
        val_loaders.extend(val_loaders_per_type)                                #collect all validation data loaders
        test_loaders.extend(test_loaders_per_type)                              #collect all test data loaders
    model = Fusion_Head(models, mode=mode)                                      #create intermediate fusion head for fused MLP models
    model = model.to(device)
    return model, train_loaders, val_loaders, test_loaders

#-------------------------------------------Main part---------------------------------------------------------
def generate_integer_combinations(num_integers, end_value):
    # Initialize the list to hold the combinations of integers
    integer_combinations = []
    # Initialize the starting array
    start_array = [1] * num_integers

    def increment_array(arr, end_value):
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] < end_value[i]:
                arr[i] += 1
                return True
            arr[i] = 1
        return False
    # Append the first array
    integer_combinations.append(start_array.copy())
    # Generate all arrays
    while increment_array(start_array, end_value):
        integer_combinations.append(start_array.copy())
    return integer_combinations


def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion):
    #create intermediate fusion head for fused MLP models
    model, train_loaders, val_loaders, _ = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, solution, mode, device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    #Training
    model_path = 'temp_Brute_force_search_best_model_min_val_loss.pth'
    print("Combination of layers: {}".format(solution))
    #train and validate fused MLP models with fusion head
    dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
    checkpoint = torch.load(model_path)
    loss = checkpoint['loss']                                           #Validation results of brute-force search
    return loss

def intermediate_fusion_brute_force_search(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion):
    # Calculate number of fused MLP models + fusion head where to optimize the number of NN layers
    num_integers = sum(1 for data_type in dimension_dict.keys() for i in loaders_dict["train"][data_type]) # Number of integers in each array
    # Upper bound of number of layers
    end_value = [int(np.log2(up_b)) for up_b in dimension_dict.values()]
    layer_combinations = generate_integer_combinations(num_integers, end_value)
    best_loss = float("inf")
    best_solution = [1] * num_integers
    best_combo = {'loss': best_loss, 'solution': best_solution}
    # calculate combination of layers by incrementing number of layers in each MLP model one by one till upper boundary
    for solution in layer_combinations:
        loss = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, mode, criterion)
        if loss < best_loss:
            best_loss = loss
            best_combo['loss'] = best_loss
            best_combo['solution'] = solution
            final_checkpoint = torch.load('temp_Brute_force_search_best_model_min_val_loss.pth')
            torch.save({'model_state_dict': final_checkpoint['model_state_dict']}, 'Brute_force_search_best_model.pth')
    # return the best combination of NN layers and its loss
    os.remove('temp_Brute_force_search_best_model_min_val_loss.pth')
    model, _, _, test_loaders = get_fusion_model_and_dataloaders(dimension_dict, loaders_dict, best_combo['solution'], mode, device)
    test_model = load_model(model, 'Brute_force_search_best_model.pth')
    test_loss = new_validate_intermediate(test_model, test_loaders, criterion, device)
    return best_combo['solution'], test_loss
