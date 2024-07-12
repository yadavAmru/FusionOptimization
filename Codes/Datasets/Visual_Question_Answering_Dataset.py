# !pip install datasets
from datasets import load_dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


#-------------------------------------------Dataset and Dataloader part-------------------------------------------------------------
class Combined_VQADataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        # Load dataset from CSV
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        # Initialize the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Preprocess the dataset
        self.preprocess_dataset()

    def preprocess_dataset(self):
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "data_train.csv"),
                "test": os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "data_eval.csv")
            }
        )

        with open(os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "answer_space.txt")) as f:
            answer_space = f.read().splitlines()

        self.data = dataset.map(
            lambda examples: {
                'label': [
                    answer_space.index(ans.replace(" ", "").split(",")[0])
                    for ans in examples['answer']
                ]
            },
            batched=True
        )

        # Convert dataset to DataFrame for easy indexing
        self.data = pd.DataFrame(self.data['train'])

    def __len__(self):
        # return len(self.data)
        return 100

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        question = row['question']
        label = row['label']

        image_path = os.path.join(self.image_folder, f"{image_id}.png")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image.view(-1)

        # Tokenize the question
        inputs = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
        # Ensure the tensors are in the correct format for the DataLoader
        input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
        input_ids = input_ids.view(-1)
        attention_mask = inputs['attention_mask'].squeeze(0)  # Remove batch dimension
        combined_data = torch.cat([image, input_ids], dim=0)

        return combined_data, label


class Image_VQADataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        # Load dataset from CSV
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        # Preprocess the dataset
        self.preprocess_dataset()

    def preprocess_dataset(self):
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "data_train.csv"),
                "test": os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "data_eval.csv")
            }
        )

        with open(os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "answer_space.txt")) as f:
            answer_space = f.read().splitlines()

        self.data = dataset.map(
            lambda examples: {
                'label': [
                    answer_space.index(ans.replace(" ", "").split(",")[0])
                    for ans in examples['answer']
                ]
            },
            batched=True
        )

        # Convert dataset to DataFrame for easy indexing
        self.data = pd.DataFrame(self.data['train'])

    def __len__(self):
        # return len(self.data)
        return 100

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = row['label']

        image_path = os.path.join(self.image_folder, f"{image_id}.png")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


class Text_VQADataset(Dataset):
    def __init__(self, csv_file):
        # Load dataset from CSV
        self.data = pd.read_csv(csv_file)
        # Initialize the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Preprocess the dataset
        self.preprocess_dataset()

    def preprocess_dataset(self):
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "data_train.csv"),
                "test": os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "data_eval.csv")
            }
        )

        with open(os.path.join("/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset", "answer_space.txt")) as f:
            answer_space = f.read().splitlines()

        self.data = dataset.map(
            lambda examples: {
                'label': [
                    answer_space.index(ans.replace(" ", "").split(",")[0])
                    for ans in examples['answer']
                ]
            },
            batched=True
        )

        # Convert dataset to DataFrame for easy indexing
        self.data = pd.DataFrame(self.data['train'])

    def __len__(self):
        # return len(self.data)
        return 100

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row['question']
        label = row['label']

        # Tokenize the question
        inputs = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
        # Ensure the tensors are in the correct format for the DataLoader
        input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = inputs['attention_mask'].squeeze(0)  # Remove batch dimension

        return input_ids, label


#--------------------------------------------------------Main part--------------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Early fusion datasets and dataloaders
batch_size = 8
n_loaders = os.cpu_count()

vqa_train_combined_dataset = Combined_VQADataset('/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/data_train.csv', '/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/images', transform)
vqa_train_combined_loader = torch.utils.data.DataLoader(vqa_train_combined_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders, pin_memory=True)
vqa_val_combined_dataset = Combined_VQADataset('/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/data_eval.csv', '/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/images', transform)
vqa_val_combined_loader = torch.utils.data.DataLoader(vqa_val_combined_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders, pin_memory=True)

#Intermediate and late fusion datasets and dataloaders
vqa_train_image_dataset = Image_VQADataset('/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/data_train.csv', '/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/images', transform)
vqa_train_image_loader = torch.utils.data.DataLoader(vqa_train_image_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders, pin_memory=True)
vqa_val_image_dataset = Image_VQADataset('/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/data_eval.csv', '/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/images', transform)
vqa_val_image_loader = torch.utils.data.DataLoader(vqa_val_image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders, pin_memory=True)

vqa_train_text_dataset = Text_VQADataset('/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/data_train.csv')
vqa_train_text_loader = torch.utils.data.DataLoader(vqa_train_text_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders, pin_memory=True)
vqa_val_text_dataset = Text_VQADataset('/content/drive/MyDrive/Colab Notebooks/Lab rotation prof. Heider/Visual-Question-Answering-Dataset/data_eval.csv')
vqa_val_text_loader = torch.utils.data.DataLoader(vqa_val_text_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders, pin_memory=True)

#----------------------------------------------Dictionary of data loaders and Dictionary of input dimensions for each data type/modality----------------------------------------------
loaders_dict_vqa = {"train": {"text": [], "image": []},
                "val": {"text": [], "image": []},
                "test": {"text": [], "image": []}}
loaders_dict_vqa["train"]["text"].append(vqa_train_text_loader)
loaders_dict_vqa["train"]["image"].append(vqa_train_image_loader)
loaders_dict_vqa["train"]["combined"] = vqa_train_combined_loader
loaders_dict_vqa["val"]["text"].append(vqa_val_text_loader)
loaders_dict_vqa["val"]["image"].append(vqa_val_image_loader)
loaders_dict_vqa["val"]["combined"] = vqa_train_combined_loader
loaders_dict_vqa["test"]["text"].append(vqa_val_text_loader)
loaders_dict_vqa["test"]["image"].append(vqa_val_image_loader)
loaders_dict_vqa["test"]["combined"] = vqa_train_combined_loader

dimension_dict_vqa = {}
text_dim = vqa_train_text_dataset[0][0].shape[0]
dimension_dict_vqa["text"] = text_dim
image_dim = vqa_train_image_dataset[0][0].shape[0] * vqa_train_image_dataset[0][0].shape[1] * vqa_train_image_dataset[0][0].shape[2]
dimension_dict_vqa["image"] = image_dim