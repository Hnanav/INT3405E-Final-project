# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

# Preparing for TPU usage
# import torch_xla
# import torch_xla.core.xla_model as xm
# device = xm.xla_device()

# Sections of config

# Defining some key variables that will be used later on in the training
MAX_LEN = 15
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 1e-02
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Creating the dataset and dataloader for the neural network
class CustomDataset(Dataset):

    """Dataset wrapper for the dataset of title, without image modality
    """

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.title = dataframe['title']
        self.targets = self.data['genre_list']
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        comment_text = str(self.title[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer(
            trainDF['title'][3204],
            None,
            add_special_tokens=True,
            max_length=30,
            padding=True,
            return_token_type_ids=True,
            truncation=True
        )

inputs['input_ids']

train = CustomDataset(trainDF, tokenizer,  30)
train[1651]

train_size = 0.8
train_dataset=trainDF.sample(frac=train_size,random_state=200)
val_dataset=trainDF.drop(train_dataset.index).reset_index(drop=True)

train_dataset = train_dataset.reset_index(drop=True)
test_dataset = testDF.reset_index(drop=True)


print("FULL Dataset: {}".format(trainDF.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VALID Dataset: {}".format(val_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
valid_set = CustomDataset(val_dataset, tokenizer, MAX_LEN)
test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
valid_loader = DataLoader(valid_set, **valid_params)
test_loader = DataLoader(test_set, **valid_params)