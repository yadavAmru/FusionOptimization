import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import numpy as np
import pygad

from Dataset import testX_images, testX_attributes, testY, img_dim, attr_dim
from Dataset import train_image_loader, train_attr_loader, val_image_loader, val_attr_loader
from late_fusion import save_model, load_model
#-----------------------------------------------------MLP model part------------------------------------------------------------------
def train_one_epoch_intermediate(model, optimizer, train_loader1, train_loader2, criterion, device):
  model.train()
  loss_step = []
  for index, data in enumerate(zip(train_loader1, train_loader2)):
    optimizer.zero_grad()
    inputs1, labels1, inputs2, labels2 = data[0][0], data[0][1], data[1][0], data[1][1]
    inputs1, labels1, inputs2, labels2 = inputs1.to(device), labels1.to(device), inputs2.to(device), labels2.to(device)
    # inputs1, labels1, inputs2, labels2 = inputs1.double(), labels1, inputs2.double(), labels2
    outputs = model(inputs1, inputs2)
    loss = criterion(outputs, labels1)
    loss.backward()
    optimizer.step()
    loss_step.append(loss.item())
  loss_curr_epoch = np.mean(loss_step)
  return loss_curr_epoch

def validate_intermediate(model, val_loader1, val_loader2, criterion, device):
  model.eval()
  loss_step = []
  for index, data in enumerate(zip(val_loader1, val_loader2)):
    inputs1, labels1, inputs2, labels2 = data[0][0], data[0][1], data[1][0], data[1][1]
    inputs1, labels1, inputs2, labels2 = inputs1.to(device), labels1.to(device), inputs2.to(device), labels2.to(device)
    # inputs1, labels1, inputs2, labels2 = inputs1.double(), labels1, inputs2.double(), labels2
    outputs = model(inputs1, inputs2)
    val_loss = criterion(outputs, labels1)
    loss_step.append(val_loss.item())
  val_loss_epoch = torch.tensor(loss_step).mean().numpy()
  return val_loss_epoch

def train_intermediate(model, optimizer, num_epochs, train_loader1, train_loader2, val_loader1, val_loader2, criterion, device):
  dict_log = {"train_loss": [], "val_loss": []}
  best_val_loss = 1e8
  device = torch.device(device)
  model = model.to(device)
  for epoch in range(num_epochs):
    loss_curr_epoch = train_one_epoch_intermediate(model, optimizer, train_loader1, train_loader2, criterion, device=device)
    val_loss = validate_intermediate(model, val_loader1, val_loader2, criterion, device)
    # Print epoch results to screen
    print(f'Ep {epoch + 1}/{num_epochs}: Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
    dict_log["train_loss"].append(loss_curr_epoch)
    dict_log["val_loss"].append(val_loss)
    # Use this code to save the model with the best validation loss
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_model(model, f'intermediate_best_model_min_val_loss.pth', epoch, optimizer, val_loss)
  return dict_log

class MLP_intermediate(nn.Module):
    def __init__(self, input_dim, n_layers = 5, output_dim=1):
        super(MLP_intermediate, self).__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers

        in_nodes = input_dim
        out_nodes = 64
        layer_list = nn.ModuleList()
        for i in range(n_layers):
            layer_list.append(nn.Linear(in_nodes, out_nodes))
            layer_list.append(nn.ReLU())
            in_nodes = out_nodes
            out_nodes = out_nodes//2
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.layers(x.float())
        return x

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, output_dim=1):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.hidden_fc = nn.Linear(list(modelA.children())[0][-2].out_features + list(modelB.children())[0][-2].out_features, 8)
        self.output_fc = nn.Linear(8, output_dim)

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)  # .detach() Detaching models, so models will not be updated
        x = F.relu(self.hidden_fc(x))
        x = self.output_fc(x)
        return x
    
def fitness_func(ga_instance, solution, solution_idx):
  global MLP_intermediate, MyEnsemble, train_intermediate, load_model, train_image_loader, train_attr_loader, val_image_loader, val_attr_loader, img_dim, attr_dim, testX_images, testX_attributes, testY
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  new_mlp_img = MLP_intermediate(input_dim=img_dim, n_layers = solution[0])
  new_mlp_num = MLP_intermediate(input_dim=attr_dim, n_layers = solution[1])
  model = MyEnsemble(new_mlp_img, new_mlp_num)

  optimizer_intermediate = optim.Adam(model.parameters(), lr=0.01)
  criterion = nn.L1Loss()
  train_intermediate(model, optimizer_intermediate, 1, train_image_loader, train_attr_loader, val_image_loader, val_attr_loader, criterion, device)
  model = load_model(model, 'intermediate_best_model_min_val_loss.pth')
  model.eval()
  intermediate_image_tensor = torch.from_numpy(testX_images).to(device)
  intermediate_attribute_tensor = torch.from_numpy(testX_attributes).to(device)
  predictions = model(intermediate_image_tensor, intermediate_attribute_tensor)
  mean_loss = np.mean(np.abs(predictions.to('cpu').detach().numpy() - testY))
  solution_fitness = 1 / (mean_loss + 1e-6)
  return solution_fitness

def callback_generation(ga_instance):
    print("Generation: {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Best solution = {fitness}".format(fitness=ga_instance.best_solution()[0]))

#-----------------------------------------------------Main part------------------------------------------------------------------
if __name__ == "__main__":
#Optimization
    ga_instance = pygad.GA(num_generations=2,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       on_generation=callback_generation,
                       sol_per_pop=2,
                       num_genes=2, # Two solutions (one for each model)
                       gene_type=int,
                       gene_space=[(1, 5), (1, 3)],  #  Range for number of layers for each model
                       )
    ga_instance.run()