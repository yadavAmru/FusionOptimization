import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

#-------------------------------------------Dataset and Dataloader part-------------------------------------------------------------
class ImageData(Dataset):                              # Class for image dataset creation with some inputed transformations
    def __init__(self, path_array, labels, transform):
        self.path_array = path_array
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.path_array.shape[0]

    def __getitem__(self, index):
        PIL_image = Image.open(self.path_array[index])
        transformed_image = self.transform(PIL_image)
        label = self.labels[index]
        return transformed_image, label

class AttrData(Dataset):                            # Class for text/numerical dataset creation
    def __init__(self, attributes, labels):
        self.attributes = attributes
        self.labels = labels

    def __len__(self):
        return self.attributes.shape[0]

    def __getitem__(self, index):
        attribute = self.attributes[index]
        label = self.labels[index]
        return attribute, label

class CombinedData(Dataset):                          # Class for creation of dataset combining all modalities(image/text/numerical data)
    def __init__(self, path_array, attributes, labels, transform):
        self.path_array = path_array
        self.attributes = attributes
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.path_array.shape[0]

    def __getitem__(self, index):
        PIL_image = Image.open(self.path_array[index])
        transformed_image = self.transform(PIL_image)
        transformed_image = transformed_image.view(-1)
        attribute = torch.from_numpy(self.attributes[index])
        transformed_attribute = attribute.view(-1)
        combined_data = torch.cat([transformed_image, transformed_attribute], dim=0)
        label = self.labels[index]
        return combined_data, label

#--------------------------------------------------------Main part--------------------------------------------------------------------

# The seven different metal oxide gas sensors, MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135 and a thermal imaging camera
gas_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Gas-Classification-Dataset/Gas Sensors Measurements/Gas_Sensors_Measurements.csv')
gas_data_folder = '/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Gas-Classification-Dataset'
for label in gas_df['Gas'].unique():
    gas_df['Corresponding Image Name'][gas_df['Corresponding Image Name'].str.contains(label)] = gas_data_folder + '/Thermal Camera Images/' + label + '/' + gas_df['Corresponding Image Name'][gas_df['Corresponding Image Name'].str.contains(label)] + '.png'


#Label encoding
gas_df['Gas'] = gas_df['Gas'].replace("NoGas", 0)
gas_df['Gas'] = gas_df['Gas'].replace("Perfume", 0)
gas_df['Gas'] = gas_df['Gas'].replace("Smoke", 1)
gas_df['Gas'] = gas_df['Gas'].replace("Mixture", 1)
gas_df = gas_df.drop(['Serial Number'], axis=1).sample(frac=1, ignore_index=True).head(100)

# Data split
train_gas_data, val_gas_data = train_test_split(gas_df, test_size=0.2, random_state=42)
train_gas_data, test_gas_data = train_test_split(train_gas_data, test_size=0.2, random_state=42)

train_gas_attr_data = train_gas_data.drop(['Corresponding Image Name'], axis=1)
train_gas_image_data = train_gas_data[['Corresponding Image Name', 'Gas']]
val_gas_attr_data = val_gas_data.drop(['Corresponding Image Name'], axis=1)
val_gas_image_data = val_gas_data[['Corresponding Image Name', 'Gas']]
test_gas_attr_data = test_gas_data.drop(['Corresponding Image Name'], axis=1)
test_gas_image_data = test_gas_data[['Corresponding Image Name', 'Gas']]

# Numerical data normalization
sc=StandardScaler()
X_train_gas_attr_data = sc.fit_transform(train_gas_attr_data.iloc[:, :-1])
y_train_gas_attr_data = train_gas_attr_data.iloc[:, -1].to_numpy()
X_val_gas_attr_data = sc.transform(val_gas_attr_data.iloc[:, :-1])
y_val_gas_attr_data = val_gas_attr_data.iloc[:, -1].to_numpy()
X_test_gas_attr_data = sc.transform(test_gas_attr_data.iloc[:, :-1])
y_test_gas_attr_data = test_gas_attr_data.iloc[:, -1].to_numpy()

#----------------------------------------------Numerical gas dataloaders----------------------------------------------
batch_size = 8
n_loaders = os.cpu_count()

train_gas_attr_dataset = AttrData(X_train_gas_attr_data, y_train_gas_attr_data)
train_gas_attr_loader = torch.utils.data.DataLoader(train_gas_attr_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_gas_attr_dataset = AttrData(X_val_gas_attr_data, y_val_gas_attr_data)
val_gas_attr_loader = torch.utils.data.DataLoader(val_gas_attr_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_gas_attr_dataset = AttrData(X_test_gas_attr_data, y_test_gas_attr_data)
test_gas_attr_loader = torch.utils.data.DataLoader(test_gas_attr_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Image gas dataloaders----------------------------------------------
mean = torch.tensor([0.5], dtype=torch.float32)
std = torch.tensor([1], dtype=torch.float32)

train_transf = transforms.Compose([                         # train image dataset transformations
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(list(mean), list(std))
])                                                          # validation/test image dataset transformations
val_transf = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(list(mean), list(std))
])

train_gas_image_dataset = ImageData(train_gas_image_data['Corresponding Image Name'].values, train_gas_image_data['Gas'].values, train_transf)
train_gas_image_loader = torch.utils.data.DataLoader(train_gas_image_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_gas_image_dataset = ImageData(val_gas_image_data['Corresponding Image Name'].values, val_gas_image_data['Gas'].values, val_transf)
val_gas_image_loader = torch.utils.data.DataLoader(val_gas_image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_gas_image_dataset = ImageData(test_gas_image_data['Corresponding Image Name'].values, test_gas_image_data['Gas'].values, val_transf)
test_gas_image_loader = torch.utils.data.DataLoader(test_gas_image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Early fusion combined data----------------------------------------------
train_gas_combined_dataset = CombinedData(train_gas_image_data['Corresponding Image Name'].values, X_train_gas_attr_data, train_gas_image_data['Gas'].values, train_transf)
train_gas_combined_loader = torch.utils.data.DataLoader(train_gas_combined_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_gas_combined_dataset = CombinedData(val_gas_image_data['Corresponding Image Name'].values, X_val_gas_attr_data, val_gas_image_data['Gas'].values, val_transf)
val_gas_combined_loader = torch.utils.data.DataLoader(val_gas_combined_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_gas_combined_dataset = CombinedData(test_gas_image_data['Corresponding Image Name'].values, X_test_gas_attr_data, test_gas_image_data['Gas'].values, val_transf)
test_gas_combined_loader = torch.utils.data.DataLoader(test_gas_combined_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Dictionary of data loaders and Dictionary of input dimensions for each data type/modality----------------------------------------------
loaders_dict_gas = {"train": {"numerical": [], "image": []},
                "val": {"numerical": [], "image": []},
                "test": {"numerical": [], "image": []}}
loaders_dict_gas["train"]["numerical"].append(train_gas_attr_loader)
loaders_dict_gas["train"]["image"].append(train_gas_image_loader)
loaders_dict_gas["train"]["combined"] = train_gas_combined_loader
loaders_dict_gas["val"]["numerical"].append(val_gas_attr_loader)
loaders_dict_gas["val"]["image"].append(val_gas_image_loader)
loaders_dict_gas["val"]["combined"] = val_gas_combined_loader
loaders_dict_gas["test"]["numerical"].append(test_gas_attr_loader)
loaders_dict_gas["test"]["image"].append(test_gas_image_loader)
loaders_dict_gas["test"]["combined"] = test_gas_combined_loader

dimension_dict_gas = {}
text_dim = train_gas_attr_dataset[0][0].shape[0]
dimension_dict_gas["numerical"] = text_dim
image_dim = train_gas_image_dataset[0][0].shape[0] * train_gas_image_dataset[0][0].shape[1] * train_gas_image_dataset[0][0].shape[2]
dimension_dict_gas["image"] = image_dim