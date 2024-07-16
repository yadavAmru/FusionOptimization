#pip install torchextractor
#pip install git+https://github.com/antoinebrl/torchextractor.git
#pip install datasets
#pip install transformers
import torchextractor as tx
from datasets import load_dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


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
    def __init__(self, image_features, attributes, labels):
        self.image_features = image_features
        self.attributes = attributes
        self.labels = labels

    def __len__(self):
        return self.image_features.shape[0]

    def __getitem__(self, index):
        image_feature = torch.from_numpy(self.image_features[index])
        transformed_image_feature = image_feature.view(-1)
        attribute = torch.from_numpy(self.attributes[index])
        transformed_attribute = attribute.view(-1)
        combined_data = torch.cat([transformed_image_feature, transformed_attribute], dim=0)
        label = self.labels[index]
        return combined_data, label

#--------------------------------------------------------Main part--------------------------------------------------------------------
dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join("/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "data_train.csv"),
                "test": os.path.join("/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "data_eval.csv")
            }
        )

with open(os.path.join("/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "answer_space.txt")) as f:
            answer_space = f.read().splitlines()

data = dataset.map(
            lambda examples: {
                'label': [
                    answer_space.index(ans.replace(" ", "").split(",")[0])
                    for ans in examples['answer']
                ]
            },
            batched=True
        )

vqa_data = pd.DataFrame(data['train']).head(1000)

# Data split
train_vqa_data, val_vqa_data = train_test_split(vqa_data, test_size=0.2, random_state=42)
train_vqa_data, test_vqa_data = train_test_split(train_vqa_data, test_size=0.2, random_state=42)

################################### TEXT preprocessing #####################################
train_questions, train_labels = train_vqa_data['question'], train_vqa_data['label']
val_questions, val_labels = val_vqa_data['question'], val_vqa_data['label']
test_questions, test_labels = test_vqa_data['question'], test_vqa_data['label']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Tokenize the questions
train_vqa_text_features = []
for i in train_questions.index.values:
    inputs = tokenizer(train_questions.loc[i], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    text_features = inputs['input_ids'].squeeze(0).cpu().numpy()
    train_vqa_text_features.append(text_features)
train_vqa_text_features = np.array(train_vqa_text_features)

val_vqa_text_features = []
for i in val_questions.index.values:
    inputs = tokenizer(val_questions.loc[i], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    text_features = inputs['input_ids'].squeeze(0).cpu().numpy()
    val_vqa_text_features.append(text_features)
val_vqa_text_features = np.array(val_vqa_text_features)

test_vqa_text_features = []
for i in test_questions.index.values:
    inputs = tokenizer(test_questions.loc[i], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    text_features = inputs['input_ids'].squeeze(0).cpu().numpy()
    test_vqa_text_features.append(text_features)
test_vqa_text_features = np.array(test_vqa_text_features)

#----------------------------------------------Text vqa dataloaders----------------------------------------------
batch_size = 64
n_loaders = os.cpu_count()

train_vqa_text_dataset = AttrData(train_vqa_text_features, train_labels.values)
train_vqa_text_loader = torch.utils.data.DataLoader(train_vqa_text_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_vqa_text_dataset = AttrData(val_vqa_text_features, val_labels.values)
val_vqa_text_loader = torch.utils.data.DataLoader(val_vqa_text_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_vqa_text_dataset = AttrData(test_vqa_text_features, test_labels.values)
test_vqa_text_loader = torch.utils.data.DataLoader(test_vqa_text_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)


#----------------------------------------------Image vqa dataloaders----------------------------------------------
# Data split
train_vqa_data, val_vqa_data = train_test_split(vqa_data, test_size=0.2, random_state=42)
train_vqa_data, test_vqa_data = train_test_split(train_vqa_data, test_size=0.2, random_state=42)

image_folder = '/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/images/'
train_vqa_image_data = image_folder + train_vqa_data['image_id'] + '.png'
val_vqa_image_data = image_folder + val_vqa_data['image_id'] + '.png'
test_vqa_image_data = image_folder + test_vqa_data['image_id'] + '.png'

train_transf = transforms.Compose([                         # train image dataset transformations
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])                                                          # validation/test image dataset transformations
val_transf = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_vqa_image_dataset = ImageData(train_vqa_image_data.values, train_labels.values, train_transf)
train_vqa_image_loader = torch.utils.data.DataLoader(train_vqa_image_dataset, batch_size=1, shuffle=True, num_workers=n_loaders)
val_vqa_image_dataset = ImageData(val_vqa_image_data.values, val_labels.values, val_transf)
val_vqa_image_loader = torch.utils.data.DataLoader(val_vqa_image_dataset, batch_size=1, shuffle=False, num_workers=n_loaders)
test_vqa_image_dataset = ImageData(test_vqa_image_data.values, test_labels.values, val_transf)
test_vqa_image_loader = torch.utils.data.DataLoader(test_vqa_image_dataset, batch_size=1, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Image features extraction----------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18 = torchvision.models.resnet18(pretrained=True)
feature_extractor = tx.Extractor(resnet18, "avgpool")
feature_extractor = feature_extractor.to(device)

train_vqa_image_features = []
for inputs, labels in train_vqa_image_loader:
    inputs = inputs.to(device)
    model_output, features = feature_extractor(inputs)
    features = list(features.values())[0].detach().cpu().numpy().squeeze()
    train_vqa_image_features.append(np.array(features))
train_vqa_image_features = np.array(train_vqa_image_features)

val_vqa_image_features = []
for inputs, labels in val_vqa_image_loader:
    inputs = inputs.to(device)
    model_output, features = feature_extractor(inputs)
    features = list(features.values())[0].detach().cpu().numpy().squeeze()
    val_vqa_image_features.append(np.array(features))
val_vqa_image_features = np.array(val_vqa_image_features)

test_vqa_image_features = []
for inputs, labels in test_vqa_image_loader:
    inputs = inputs.to(device)
    model_output, features = feature_extractor(inputs)
    features = list(features.values())[0].detach().cpu().numpy().squeeze()
    test_vqa_image_features.append(np.array(features))
test_vqa_image_features = np.array(test_vqa_image_features)

train_vqa_image_dataset = AttrData(train_vqa_image_features, train_labels.values)
train_vqa_image_loader = torch.utils.data.DataLoader(train_vqa_image_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_vqa_image_dataset = AttrData(val_vqa_image_features, val_labels.values)
val_vqa_image_loader = torch.utils.data.DataLoader(val_vqa_image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_vqa_image_dataset = AttrData(test_vqa_image_features, test_labels.values)
test_vqa_image_loader = torch.utils.data.DataLoader(test_vqa_image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Early fusion combined data----------------------------------------------
train_vqa_combined_dataset = CombinedData(train_vqa_image_features, train_vqa_text_features, train_labels.values)
train_vqa_combined_loader = torch.utils.data.DataLoader(train_vqa_combined_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_vqa_combined_dataset = CombinedData(val_vqa_image_features, val_vqa_text_features, val_labels.values)
val_vqa_combined_loader = torch.utils.data.DataLoader(val_vqa_combined_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_vqa_combined_dataset = CombinedData(test_vqa_image_features, test_vqa_text_features, test_labels.values)
test_vqa_combined_loader = torch.utils.data.DataLoader(test_vqa_combined_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Dictionary of data loaders and Dictionary of input dimensions for each data type/modality----------------------------------------------
loaders_dict_vqa = {"train": {"text": [], "image": []},
                "val": {"text": [], "image": []},
                "test": {"text": [], "image": []}}
loaders_dict_vqa["train"]["text"].append(train_vqa_text_loader)
loaders_dict_vqa["train"]["image"].append(train_vqa_image_loader)
loaders_dict_vqa["train"]["combined"] = train_vqa_combined_loader
loaders_dict_vqa["val"]["text"].append(val_vqa_text_loader)
loaders_dict_vqa["val"]["image"].append(val_vqa_image_loader)
loaders_dict_vqa["val"]["combined"] = val_vqa_combined_loader
loaders_dict_vqa["test"]["text"].append(val_vqa_text_loader)
loaders_dict_vqa["test"]["image"].append(val_vqa_image_loader)
loaders_dict_vqa["test"]["combined"] = val_vqa_combined_loader

dimension_dict_vqa = {}
text_dim = train_vqa_text_dataset[0][0].shape[0]
dimension_dict_vqa["text"] = text_dim
image_dim = train_vqa_image_dataset[0][0].shape[0]
dimension_dict_vqa["image"] = image_dim
