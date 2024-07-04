import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
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
from sklearn import preprocessing

batch_size = 2
n_loaders = os.cpu_count()
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

train_df = pd.read_csv('/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/shopee-product-matching/cleaned_train_all.csv')
train_image_path='/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/shopee-product-matching/train_images_new/'

train_df['image_path'] = train_df['image'].apply(lambda x: train_image_path + str(x))
labels_list = train_df['label_group'].value_counts().index[:100].tolist() ##filtering only 100 labels
train_df = train_df[train_df['label_group'].isin(labels_list)].reset_index(drop=True)
le = preprocessing.LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label_group'])

train_shopee_data, val_shopee_data = train_test_split(train_df, test_size=0.2, random_state=42)
train_shopee_data, test_shopee_data = train_test_split(train_shopee_data, test_size=0.2, random_state=42)

#----------------------------------------------Image shopee dataloaders----------------------------------------------
train_shopee_image_dataset = ImageData(train_shopee_data['image_path'].values, train_shopee_data['label'].values, train_transf)
val_shopee_image_dataset = ImageData(val_shopee_data['image_path'].values, val_shopee_data['label'].values, val_transf)
test_shopee_image_dataset = ImageData(test_shopee_data['image_path'].values, test_shopee_data['label'].values, val_transf)
train_shopee_image_loader = torch.utils.data.DataLoader(train_shopee_image_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_shopee_image_loader = torch.utils.data.DataLoader(val_shopee_image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_shopee_image_loader = torch.utils.data.DataLoader(test_shopee_image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Text shopee dataloaders----------------------------------------------
y_train = train_shopee_data['label'].values
y_val = val_shopee_data['label'].values
y_test = test_shopee_data['label'].values

# Transform text to vectors
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

# Step 1: Text Preprocessing (example - lowercase)
train_shopee_text_data = train_shopee_data['title'].str.lower()
val_shopee_text_data = val_shopee_data['title'].str.lower()
test_shopee_text_data = test_shopee_data['title'].str.lower()
# Step 2: Tokenization
train_shopee_text_tokenized_data = [sentence.split() for sentence in train_shopee_text_data]
val_shopee_text_tokenized_data = [sentence.split() for sentence in val_shopee_text_data]
test_shopee_text_tokenized_data = [sentence.split() for sentence in test_shopee_text_data]
# Step 3: Train Word2Vec model
word2vec_model = Word2Vec(sentences=train_shopee_text_tokenized_data, vector_size=20, window=5, min_count=1, workers=n_loaders)
# Function to convert words to vectors
def sentence_to_vector(tokenized_sentence, model, vector_size):
    vector = []
    for word in tokenized_sentence:
        if word in model.wv:
            vector.append(model.wv[word])
        else:
            vector.append(np.zeros(vector_size))  # Handle words not in vocabulary
    return vector

# Step 4: Convert tokenized sentences to vectors
vector_size = word2vec_model.vector_size
train_shopee_text_data_sequences = [sentence_to_vector(sentence, word2vec_model, vector_size) for sentence in train_shopee_text_tokenized_data]
val_shopee_text_sequences = [sentence_to_vector(sentence, word2vec_model, vector_size) for sentence in val_shopee_text_tokenized_data]
test_shopee_text_sequences = [sentence_to_vector(sentence, word2vec_model, vector_size) for sentence in test_shopee_text_tokenized_data]
# Step 5: Padding/Truncating sequences to the same length
max_sequence_length = max(max(len(seq) for seq in train_shopee_text_data_sequences), max(len(seq) for seq in val_shopee_text_sequences), max(len(seq) for seq in test_shopee_text_sequences))
train_shopee_text_data = pad_sequences(train_shopee_text_data_sequences, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
val_shopee_text_data = pad_sequences(val_shopee_text_sequences, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
test_shopee_text_data = pad_sequences(test_shopee_text_sequences, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')


train_shopee_text_dataset = AttrData(train_shopee_text_data, y_train)
val_shopee_text_dataset = AttrData(val_shopee_text_data, y_val)
test_shopee_text_dataset = AttrData(test_shopee_text_data, y_test)
train_shopee_text_loader = torch.utils.data.DataLoader(train_shopee_text_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_shopee_text_loader = torch.utils.data.DataLoader(val_shopee_text_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_shopee_text_loader = torch.utils.data.DataLoader(test_shopee_text_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Early fusion combined data----------------------------------------------
train_shopee_combined_dataset = CombinedData(train_shopee_data['image_path'].values, train_shopee_text_data, train_shopee_data['label'].values, train_transf)
train_shopee_combined_loader = torch.utils.data.DataLoader(train_shopee_combined_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_shopee_combined_dataset = CombinedData(val_shopee_data['image_path'].values, val_shopee_text_data, val_shopee_data['label'].values, val_transf)
val_shopee_combined_loader = torch.utils.data.DataLoader(val_shopee_combined_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
test_shopee_combined_dataset = CombinedData(test_shopee_data['image_path'].values, test_shopee_text_data, test_shopee_data['label'].values, val_transf)
test_shopee_combined_loader = torch.utils.data.DataLoader(test_shopee_combined_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Dictionary of data loaders and Dictionary of input dimensions for each data type/modality----------------------------------------------
loaders_dict = {"train": {"text": [], "image": []},
                "val": {"text": [], "image": []},
                "test": {"text": [], "image": []}}
loaders_dict["train"]["text"].append(train_shopee_text_loader)
loaders_dict["train"]["image"].append(train_shopee_image_loader)
loaders_dict["train"]["combined"] = train_shopee_combined_loader
loaders_dict["val"]["text"].append(val_shopee_text_loader)
loaders_dict["val"]["image"].append(val_shopee_image_loader)
loaders_dict["val"]["combined"] = val_shopee_combined_loader
loaders_dict["test"]["text"].append(test_shopee_text_loader)
loaders_dict["test"]["image"].append(test_shopee_image_loader)
loaders_dict["test"]["combined"] = test_shopee_combined_loader

dimension_dict = {}
text_dim = train_shopee_text_dataset[0][0].shape[0] * train_shopee_text_dataset[0][0].shape[1]
dimension_dict["text"] = text_dim
image_dim = train_shopee_image_dataset[0][0].shape[0] * train_shopee_image_dataset[0][0].shape[1] * train_shopee_image_dataset[0][0].shape[2]
dimension_dict["image"] = image_dim
