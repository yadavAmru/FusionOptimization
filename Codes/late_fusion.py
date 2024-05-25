import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from Dataset import testX_images, testX_attributes, testY, img_dim, attr_dim
from Dataset import train_image_loader, train_attr_loader, val_image_loader, val_attr_loader
#-----------------------------------------------------MLP model part------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.input_fc = nn.Linear(input_dim, 8)
        self.hidden_fc = nn.Linear(8, 4)
        self.output_fc = nn.Linear(4, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h_1 = F.relu(self.input_fc(x.float()))   #.float()
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred
    
def train_one_epoch(model, optimizer, train_loader, criterion, device):
  model.train()
  loss_step = []
  for inputs, labels in train_loader:
    optimizer.zero_grad()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    loss_step.append(loss.item())
  loss_curr_epoch = np.mean(loss_step)
  return loss_curr_epoch

def validate(model, val_loader, criterion, device):
  model.eval()
  loss_step = []
  for inputs, labels in val_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    val_loss = criterion(outputs, labels)
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
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model

def train(model, optimizer, num_epochs, train_loader, val_loader, criterion, device, path):
  dict_log = {"train_loss": [], "val_loss": []}
  best_val_loss = 1e8
  model = model.to(device)
  for epoch in range(num_epochs):
    loss_curr_epoch = train_one_epoch(model, optimizer, train_loader, criterion, device=device)
    val_loss = validate(model, val_loader, criterion, device)
    # Print epoch results to screen
    print(f'Ep {epoch + 1}/{num_epochs}: Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
    dict_log["train_loss"].append(loss_curr_epoch)
    dict_log["val_loss"].append(val_loss)
    # Use this code to save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, f'{path}', epoch, optimizer, val_loss)
  return dict_log

#-----------------------------------------------------Main part------------------------------------------------------------------
if __name__ == "__main__":
    mlp_num = MLP(input_dim=attr_dim)
    mlp_img = MLP(input_dim=img_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer1 = optim.Adam(mlp_num.parameters(), lr=0.01)
    optimizer2 = optim.Adam(mlp_img.parameters(), lr=0.01)
    criterion = nn.L1Loss()
#Training
    mlp_num_path = 'best_num_late_fusion_model_min_val_loss.pth'
    mlp_img_path = 'best_img_late_fusion_model_min_val_loss.pth'
    num_dict_log = train(mlp_num, optimizer1, 1, train_attr_loader, val_attr_loader, criterion, device, mlp_num_path)
    img_dict_log = train(mlp_img, optimizer2, 1, train_image_loader, val_image_loader, criterion, device, mlp_img_path)
#Testing
    model1 = load_model(mlp_num, mlp_num_path)
    model1.eval()
    attr_tensor = torch.from_numpy(testX_attributes)
    attr_tensor = attr_tensor.to(device)
    attr_preds = model1(attr_tensor)

    model2 = load_model(mlp_img, mlp_img_path)
    model2.eval()
    image_tensor = torch.from_numpy(testX_images)
    image_tensor = image_tensor.to(device)
    image_preds = model2(image_tensor)

#-----------------------------------------------------Printing results------------------------------------------------------------------
    print("-------------------------------------Results-----------------------------------------------")
    diff = (attr_preds.to('cpu').detach().numpy() + image_preds.to('cpu').detach().numpy())/2 - testY
    percentDiff = (diff / testY) * 100
    absPercentDiff = np.abs(percentDiff)
    mean = np.mean(absPercentDiff)
    print("Mean: {:.2f}%".format(mean))
    print("-------------------------------------------------------------------------------------------")