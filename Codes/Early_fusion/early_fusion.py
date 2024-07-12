import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#-----------------------------------------------------Definition of functions-----------------------------------------------------
#MLP model flexible structure
class MLP(nn.Module):
    def __init__(self, input_dim, fusion, mode, n_layers=None, output_dim=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fusion = fusion
        self.mode = mode
        self.output_dim = output_dim
        layer_list = nn.ModuleList()
        if n_layers == None:              # if we do not receive the number of layers -> halving (output nodes = input nodes / 2) of the nodes for each neural network layer starts
            in_nodes = input_dim
            if input_dim == 0:                                  # we cannot have dimension = 0
                raise ValueError("Input dimension cannot be zero!!!")
            elif input_dim == 1 or input_dim == 2 or input_dim == 3:   # if input dimension = 1 or 2 or 3 -> we create just 1 linear layer with output dimension = 2
                out_nodes = 2
                layer_list.append(nn.Linear(in_nodes, out_nodes))
                layer_list.append(nn.ReLU())
            while (in_nodes // 2) != 0 and (in_nodes // 2) != 1 and in_nodes != 1:               # we create layers with output nodes twice less than input nodes until we reach
                out_nodes = in_nodes//2
                layer_list.append(nn.Linear(in_nodes, out_nodes))
                layer_list.append(nn.ReLU())
                in_nodes = out_nodes
            self.layers = nn.Sequential(*layer_list)
            self.out_nodes = out_nodes
        else:                                 # if we receive the number of layers -> we linearly decrease number of nodes for each neural network layer
            nodes = np.linspace(input_dim, 1, num=n_layers + 2, dtype=int)
            for idx in range(len(nodes) - 2):
                nodes[idx], nodes[idx + 1]
                layer_list.append(nn.Linear(nodes[idx], nodes[idx + 1]))
                layer_list.append(nn.ReLU())
            self.layers = nn.Sequential(*layer_list)
            self.out_nodes = nodes[-2]

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.layers(x.float())
        if self.fusion == "intermediate":       # for intermediate fusion we return extracted features before the output layer
            return x
        elif self.fusion == "early" or self.fusion ==  "late":        # for early and late fusion we return MLP model with output linear layer
            output_fc = nn.Linear(self.out_nodes, self.output_dim).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            y_pred = output_fc(x)
            if self.mode == "classification":       # for classification we need to add sigmoid activation function to the output of linear layer
                sigmoid = nn.Sigmoid().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                sigmoid_output = sigmoid(y_pred)
                return sigmoid_output
            elif self.mode == "regression":         # for regression we just return the output of linear layer
                return y_pred
            else:
                raise ValueError("Incorrect model type selected!!!")
        else:
            raise ValueError("Incorrect fusion selected!!!")

def train_one_epoch(model, optimizer, train_loader, criterion, device):
  model.train()
  loss_step = []
  for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)                                   #Forward propagation
    loss = criterion(outputs, labels.float().unsqueeze(1))    #Compute loss function
    optimizer.zero_grad()
    loss.backward()                                #Backward propagation
    optimizer.step()                               #Update parameters
    loss_step.append(loss.item())
  loss_curr_epoch = np.mean(loss_step)
  return loss_curr_epoch

def validate(model, val_loader, criterion, device):
  model.eval()
  loss_step = []
  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)                                         #Forward propagation
      val_loss = criterion(outputs, labels.float().unsqueeze(1))      #Compute loss function
      loss_step.append(val_loss.item())
  val_loss_epoch = torch.tensor(loss_step).mean().numpy()
  return val_loss_epoch

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

def train(model, optimizer, num_epochs, train_loader, val_loader, criterion, device, path):
  dict_log = {"train_loss": [], "val_loss": []}
  best_val_loss = float("inf")
  # model = model.to(device)
  for epoch in range(num_epochs):
    loss_curr_epoch = train_one_epoch(model, optimizer, train_loader, criterion, device=device)       #train for one epoch
    val_loss = validate(model, val_loader, criterion, device)                                         #validate in the same epoch
    # Print epoch results to screen
    print(f'Ep {epoch + 1}/{num_epochs}: Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
    dict_log["train_loss"].append(loss_curr_epoch)
    dict_log["val_loss"].append(val_loss)
    # Use this code to save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, f'{path}', epoch, optimizer, val_loss)
  return dict_log

#----------------------------------------------------Early fusion---------------------------------------------------------------
def early_fusion(dimension_dict, loaders_dict, device, lr, num_epochs, mode, criterion):
    input_dim = sum(list(dimension_dict.values()))
    model = MLP(input_dim=input_dim, fusion="early", mode=mode)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model_path = 'best_early_fusion_model.pth'
    #train model with validation
    dict_log = train(model, optimizer, num_epochs, loaders_dict['train']['combined'], loaders_dict['val']['combined'], criterion, device, model_path)
    # checkpoint = torch.load('best_early_fusion_model.pth')
    # loss = checkpoint['loss']                                                   #Validation results of early fusion
    test_model = load_model(model, model_path)
    test_loss = validate(test_model, loaders_dict['test']['combined'], criterion, device)   #Test results of early fusion
    return test_loss, dict_log
