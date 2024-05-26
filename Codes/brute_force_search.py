import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
from intermediate_fusion_GA import MLP_intermediate, MyEnsemble, train_intermediate
from late_fusion import save_model, load_model

#-------------------------------------------Definition of functions---------------------------------------------------------
def brute_force_search(attr_dim, img_dim, train_attr_loader, train_image_loader, val_attr_loader, val_image_loader, device, lr, num_epochs, criterion): 
  best_loss = 1e8
  solution0, solution1 = 0, 0
  for i in range(1, 5):
    for j in range(1, 5):
      print("Combination of layers: {}".format([i, j]))
      new_mlp_img = MLP_intermediate(input_dim=img_dim, n_layers=i)
      new_mlp_num = MLP_intermediate(input_dim=attr_dim, n_layers=j)
      model = MyEnsemble(new_mlp_img, new_mlp_num)
      optimizer_intermediate = optim.Adam(model.parameters(), lr=lr)
      path = 'intermediate_best_model_min_val_loss.pth'
      train_intermediate(model, optimizer_intermediate, num_epochs, train_image_loader, train_attr_loader, val_image_loader, val_attr_loader, criterion, device, path)
      best_model = load_model(model, 'intermediate_best_model_min_val_loss.pth')
      checkpoint = torch.load('intermediate_best_model_min_val_loss.pth')
      loss = checkpoint['loss']
      if loss < best_loss:
        solution0, solution1 = i, j
        best_loss = loss
        save_model(best_model, f'brute_force_best_model.pth', 1, optimizer_intermediate, loss)
  brute_force_checkpoint = torch.load('brute_force_best_model.pth')
  brute_force_loss = brute_force_checkpoint['loss']
  return brute_force_loss, solution0, solution1