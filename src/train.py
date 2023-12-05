import os

import pandas as pd
import torch
import torch.optim as opt
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from dataset import prepare_data, prepare_dataset
from model.MLP import TextClassificationModel
from model.utils import evaluate, train
from utils.representation import build_vocabulary

folder_paths = {
    "train": "data/data_train/train",
    "valid": "data/data_train/test",
    "test": "data/data_test/test",
}


dataset_dir = "./dataset"

if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
    train_df = prepare_data(folder_paths["train"])
    valid_df = prepare_data(folder_paths["valid"])
    test_df = prepare_data(folder_paths["test"])

    train_df.to_csv(os.path.join(dataset_dir, "train_df.csv"))
    valid_df.to_csv(os.path.join(dataset_dir, "valid_df.csv"))
    test_df.to_csv(os.path.join(dataset_dir, "test_df.csv"))
train_df = pd.read_csv(os.path.join(dataset_dir, "train_df.csv"))
valid_df = pd.read_csv(os.path.join(dataset_dir, "valid_df.csv"))
test_df = pd.read_csv(os.path.join(dataset_dir, "test_df.csv"))
tokenizer = get_tokenizer("basic_english")
vocabulary = build_vocabulary(train_df["sentence"], tokenizer)

train_dataset = prepare_dataset(train_df, vocabulary, tokenizer)
valid_dataset = prepare_dataset(valid_df, vocabulary, tokenizer)
test_dataset = prepare_dataset(test_df, vocabulary, tokenizer)


# Create collate_batch
def collate_batch(
    batch, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    encoded_sentences, labels, offsets = [], [], [0]
    for encoded_sentence, label in batch:
        labels.append(label)
        encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.int64)
        encoded_sentences.append(encoded_sentence)
        offsets.append(encoded_sentence.size(0))

    labels = torch.tensor(labels, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    encoded_sentences = torch.cat(encoded_sentences)
    return encoded_sentences.to(device), offsets.to(device), labels.to(device)


train_loader = DataLoader(
    list(train_dataset),
    batch_size=8,
    shuffle=True,
    collate_fn=collate_batch,
    drop_last=True,
)
valid_loader = DataLoader(
    list(valid_dataset), batch_size=8, shuffle=False, drop_last=False
)

test_loader = DataLoader(
    list(test_dataset), batch_size=8, shuffle=False, drop_last=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_class = len(set(train_df["label"]))
vocab_size = len(vocabulary)
embed_dim = 100
model = TextClassificationModel(vocab_size, embed_dim, num_class).to(device)
optimizer = opt.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
epoch_acc, epoch_loss = train(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_loader,
    epochs=500,
)

print(epoch_acc, epoch_loss)
